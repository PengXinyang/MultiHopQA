# 多跳问答并行执行辅助逻辑。

import copy
import logging
import math
import multiprocessing as mp
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import utils
from graph_of_thoughts import language_models, operations
from graph_of_thoughts.language_models.gemini_grouped_failover import (
    detect_gemini_parallel_num_groups,
    parallel_gemini_groups_enabled,
)
from multi_hop_graphs import io, cot, tot, got, multiAgentGoT
from multi_hop_parser import MultiHopParser
from multi_hop_prompter import MultiHopPrompter
from role_aware_lm import RoleAwareLM


# 与 multi_hop_graphs.multiAgentGoT 默认形参一致；并行子进程与串行调用均使用此处（改图规模请改 graphs 或本常量）。
MAGOT_DEFAULTS: Dict[str, int] = {
    "got_hops": 4,
    "got_branch_k": 2,
    "got_critic_retries": 3,
}

# 仅并行池 initializer 在子进程中递增；主进程保持 None
_PARALLEL_GEMINI_SLOT_RAW: Optional[int] = None


def _pool_init_gemini_slot(counter) -> None:
    global _PARALLEL_GEMINI_SLOT_RAW
    with counter.get_lock():
        s = int(counter.value)
        counter.value = s + 1
    _PARALLEL_GEMINI_SLOT_RAW = s


def _effective_gemini_parallel_group(config_lm_path: str) -> Optional[int]:
    """子进程内：worker 序号对组数取模，得到 0..num_groups-1。"""
    if _PARALLEL_GEMINI_SLOT_RAW is None:
        return None
    n = detect_gemini_parallel_num_groups(config_lm_path)
    return int(_PARALLEL_GEMINI_SLOT_RAW) % max(1, n)


def make_lm_for_method(
    method: Callable[..., operations.GraphOfOperations],
    role_model_names: Dict[str, str],
    config_lm_path: str,
) -> Any:
    """按方法类型构造 LM：multiAgentGoT* 用多角色 RoleAwareLM，其余用 default 单模型。"""
    gp = _effective_gemini_parallel_group(config_lm_path)
    if method.__name__.startswith("multiAgentGoT"):
        role_to_lm = {}
        for role, model_name in role_model_names.items():
            if model_name == "__lite__":
                role_to_lm[role] = language_models.LightweightModelGroup(
                    config_lm_path,
                    cache=True,
                    retries_per_model=3,
                    gemini_parallel_group_0based=gp,
                )
            elif model_name == "__heavy__":
                role_to_lm[role] = language_models.HeavyModelGroup(
                    config_lm_path,
                    cache=True,
                    retries_per_model=3,
                    gemini_parallel_group_0based=gp,
                )
            else:
                role_to_lm[role] = language_models.build_language_model(
                    config_lm_path,
                    model_name=model_name,
                    cache=True,
                    gemini_parallel_group_0based=gp,
                )
        return RoleAwareLM(role_to_lm=role_to_lm, default_role="default")
    model_name = role_model_names["default"]
    if model_name == "__lite__":
        return language_models.LightweightModelGroup(
            config_lm_path,
            cache=True,
            retries_per_model=3,
            gemini_parallel_group_0based=gp,
        )
    if model_name == "__heavy__":
        return language_models.HeavyModelGroup(
            config_lm_path,
            cache=True,
            retries_per_model=3,
            gemini_parallel_group_0based=gp,
        )
    return language_models.build_language_model(
        config_lm_path,
        model_name=model_name,
        cache=True,
        gemini_parallel_group_0based=gp,
    )


def method_to_parallel_tag(method: Callable[..., operations.GraphOfOperations]) -> str:
    """将主进程中的方法对象转为可 pickle 的短标签；multiAgentGoT* 均映射为 multiAgentGoT。"""
    n = method.__name__
    if n.startswith("multiAgentGoT"):
        return "multiAgentGoT"
    if n in ("io", "cot", "tot", "got"):
        return n
    raise ValueError(
        f"并行模式下无法序列化方法 {n!r}；请使用 io/cot/tot/got 或名称以 multiAgentGoT 开头的图工厂。"
    )


def _resolve_method_from_tag(
    tag: str, pg: Dict[str, int]
) -> Callable[..., operations.GraphOfOperations]:
    """子进程内根据标签还原与主进程等价的图构建函数。"""
    if tag == "multiAgentGoT":

        def factory(max_subquestions: int = 4) -> operations.GraphOfOperations:
            nh = max(
                1,
                int(max_subquestions or 1),
                int(pg.get("got_hops", MAGOT_DEFAULTS["got_hops"])),
            )
            return multiAgentGoT(
                nh,
                int(pg.get("got_branch_k", MAGOT_DEFAULTS["got_branch_k"])),
                max(1, int(pg.get("got_critic_retries", MAGOT_DEFAULTS["got_critic_retries"]))),
            )

        factory.__name__ = "multiAgentGoT"
        return factory
    table = {"io": io, "cot": cot, "tot": tot, "got": got}
    if tag not in table:
        raise ValueError(f"未知 method 标签: {tag!r}")
    return table[tag]


def _multi_hop_pool_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    子进程入口：对单题按 method_tags 顺序串行跑多种方法（如 cot → tot → got → multiAgentGoT）。
    必须定义为模块顶层函数以便 Windows spawn 下可 pickle。
    """
    item = copy.deepcopy(payload["item"])
    run_dir = payload["run_dir"]
    config_lm_path = payload["config_lm_path"]
    role_model_names: Dict[str, str] = payload["role_model_names"]
    method_tags: List[str] = list(payload["method_tags"])
    pg = {
        "got_hops": int(payload.get("got_hops", MAGOT_DEFAULTS["got_hops"])),
        "got_branch_k": int(payload.get("got_branch_k", MAGOT_DEFAULTS["got_branch_k"])),
        "got_critic_retries": int(
            payload.get("got_critic_retries", MAGOT_DEFAULTS["got_critic_retries"])
        ),
    }

    prompter = MultiHopPrompter()
    parser = MultiHopParser()
    total_cost = 0.0

    try:
        for tag in method_tags:
            method = _resolve_method_from_tag(tag, pg)
            lm = make_lm_for_method(method, role_model_names, config_lm_path)
            c = utils.runSingleMethod(
                item=item,
                method=method,
                lm=lm,
                prompter=prompter,
                parser=parser,
                run_dir=run_dir,
                event_sink=None,
            )
            total_cost += float(c)
        return {
            "cost": float(total_cost),
            "_id": str(item.get("_id", "")),
            "ok": True,
            "error": None,
        }
    except Exception as e:
        logging.error("Pool worker failed _id=%s: %s", item.get("_id", ""), e)
        return {
            "cost": float(total_cost),
            "_id": str(item.get("_id", "")),
            "ok": False,
            "error": f"{e!s}\n{traceback.format_exc()}",
        }


def run_parallel_methods(
    selected: List[Dict[str, Any]],
    run_dir: str,
    config_lm_path: str,
    role_model_names: Dict[str, str],
    method_tags: List[str],
    parallel_workers: int,
    budget: float,
) -> float:
    """按“每题一个进程、题内方法串行”的策略运行多跳问答任务。"""
    pg = dict(MAGOT_DEFAULTS)
    tasks: List[Dict[str, Any]] = []
    for item in selected:
        tasks.append(
            {
                "item": item,
                "run_dir": run_dir,
                "config_lm_path": config_lm_path,
                "role_model_names": dict(role_model_names),
                "method_tags": method_tags,
                "got_hops": int(pg["got_hops"]),
                "got_branch_k": int(pg["got_branch_k"]),
                "got_critic_retries": int(pg["got_critic_retries"]),
            }
        )

    spent = 0.0
    total = len(tasks)
    done = 0
    use_pool_gemini = parallel_gemini_groups_enabled(config_lm_path)
    ctx = mp.get_context("spawn")
    pool_kw: Dict[str, Any] = {"max_workers": parallel_workers, "mp_context": ctx}
    if use_pool_gemini:
        pool_kw["initializer"] = _pool_init_gemini_slot
        pool_kw["initargs"] = (ctx.Value("i", 0),)
    ex = ProcessPoolExecutor(**pool_kw)
    try:
        futures = {ex.submit(_multi_hop_pool_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            c = float(res.get("cost") or 0.0)
            spent += c
            budget -= c
            sid = res.get("_id", "")
            chain = "-".join(method_tags)
            if res.get("ok"):
                print(
                    f"  [并行 {done}/{total}] 完成 _id={sid} "
                    f"串行[{chain}] 合计 cost=${c:.4f}"
                )
            else:
                print(f"  [并行 {done}/{total}] 失败 _id={sid} err={res.get('error', '')[:200]}")
            if math.isfinite(budget) and budget <= 0:
                logging.error("预算耗尽；已提交的任务仍会由子进程跑完，请提高 budget 或减少样本。")
            n_chk = int(utils.DATASET_AGGREGATE_CHECKPOINT_EVERY)
            if n_chk > 0 and done % n_chk == 0:
                utils.finalize_run_aggregate(
                    run_dir,
                    progress_completed_n=done,
                    print_table=False,
                )
    except KeyboardInterrupt:
        print("\n[!] 收到中断信号 (Ctrl+C)，正在强制终止所有子进程...")
        for p in ex._processes.values():
            p.terminate()
        ex.shutdown(wait=False)
        print("[!] 子进程已终止，退出程序。")
        sys.exit(1)
    finally:
        ex.shutdown(wait=True)
    utils.finalize_run_aggregate(run_dir)
    return spent
