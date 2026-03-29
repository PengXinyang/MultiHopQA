# 多跳推理示例：在 GoT 框架下对多文档问答进行 IO / CoT / ToT / GoT 多种推理方式。

import argparse
import copy
import logging
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import utils
from graph_of_thoughts import language_models, operations
from graph_of_thoughts.visualization import EventStore, start_realtime_server
from multi_hop_graphs import io, cot, tot, got, multiAgentGoT
from multi_hop_parser import MultiHopParser
from multi_hop_prompter import MultiHopPrompter
from role_aware_lm import RoleAwareLM


def _make_lm_for_method(
    method: Callable[..., operations.GraphOfOperations],
    role_model_names: Dict[str, str],
    config_lm_path: str,
) -> Any:
    """按方法类型构造 LM：multiAgentGoT* 用多角色 RoleAwareLM，其余用 default 单模型。"""
    if method.__name__.startswith("multiAgentGoT"):
        role_to_lm = {}
        for role, model_name in role_model_names.items():
            if model_name == "__lite__":
                role_to_lm[role] = language_models.LightweightModelGroup(
                    config_lm_path, cache=True, retries_per_model=3
                )
            elif model_name == "__heavy__":
                role_to_lm[role] = language_models.HeavyModelGroup(
                    config_lm_path, cache=True, retries_per_model=3
                )
            else:
                role_to_lm[role] = language_models.build_language_model(
                    config_lm_path,
                    model_name=model_name,
                    cache=True,
                )
        return RoleAwareLM(role_to_lm=role_to_lm, default_role="default")
    model_name = role_model_names["default"]
    if model_name == "__lite__":
        return language_models.LightweightModelGroup(
            config_lm_path, cache=True, retries_per_model=3
        )
    if model_name == "__heavy__":
        return language_models.HeavyModelGroup(
            config_lm_path, cache=True, retries_per_model=3
        )
    return language_models.build_language_model(
        config_lm_path,
        model_name=model_name,
        cache=True,
    )


def _method_to_parallel_tag(method: Callable[..., operations.GraphOfOperations]) -> str:
    """
    将主进程中的方法对象转为可 pickle 的短标签，供子进程还原。
    自定义 multiAgentGoT 工厂（如 multiAgentGoT_configured）一律映射为 multiAgentGoT，具体超参由 parallel_got 传入子进程。
    """
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
            nh = max(1, int(max_subquestions or 1), int(pg.get("got_hops", 4)))
            return multiAgentGoT(
                nh,
                int(pg.get("got_branch_k", 2)),
                max(1, int(pg.get("got_critic_retries", 3))),
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
        "got_hops": int(payload.get("got_hops", 4)),
        "got_branch_k": int(payload.get("got_branch_k", 2)),
        "got_critic_retries": int(payload.get("got_critic_retries", 3)),
    }

    prompter = MultiHopPrompter()
    parser = MultiHopParser()
    total_cost = 0.0

    try:
        for tag in method_tags:
            method = _resolve_method_from_tag(tag, pg)
            lm = _make_lm_for_method(method, role_model_names, config_lm_path)
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


# --- Run ---


def run(
        data_ids: List[int],
        methods: List[Callable[..., operations.GraphOfOperations]],
        budget: float,
        role_model_names: dict[str, str],
        dataset: str,
        data_path: str = None,
        max_samples: int = 100,
        vis_store: Optional[EventStore] = None,
        parallel_workers: int = 1,
        parallel_got: Optional[dict] = None,
) -> float:
    """
    加载多跳问答数据集（HotpotQA / MuSiQue），对指定样本和指定方法运行 GoT 框架，
    并将每次运行的 GRS（Graph Reasoning State）输出到 `results/` 目录下。
    
    Args:
        data_ids: 要运行的样本索引列表，None 或空列表表示全部
        methods: 方法列表，每个方法返回一个 GraphOfOperations
        budget: 预算（美元），超出后停止
        role_model_names: 语言模型名称
        dataset: 运行哪个数据集
        data_path: 数据集路径，None 则使用默认 HotpotQA
        max_samples: 最大加载样本数
        parallel_workers: >1 时按「每题一个进程」并行；同一题内 methods 顺序串行执行
        parallel_got: 并行时 multiAgentGoT 使用的 got_hops / got_branch_k / got_critic_retries
    
    Returns:
        spent: 实际花费（美元）
    """
    # 1. 解析数据路径
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "hotpotQA", "hotpot_dev_distractor_v1.json"
        )
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))

    # 2. 加载数据
    data = utils.loadMultiHopData(data_path, max_samples=max_samples)
    if not data:
        raise FileNotFoundError(f"No data loaded from {data_path}")

    # 3. 选择样本
    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected = [data[i] for i in data_ids if i < len(data)]

    pw = max(1, int(parallel_workers or 1))
    parallel_method_tags: Optional[List[str]] = None
    if pw > 1:
        try:
            parallel_method_tags = [_method_to_parallel_tag(m) for m in methods]
        except ValueError as e:
            raise ValueError(
                "parallel_workers>1 时，methods 须为可序列化标签："
                "io / cot / tot / got，以及任意名称以 multiAgentGoT 开头的工厂。"
            ) from e

    # 4. 创建运行目录和配置
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    config_extra: Dict[str, Any] = {
        "data_path": data_path,
        "data_ids": data_ids[:len(selected)],
        "budget": budget,
        "max_samples": max_samples,
        "parallel_workers": pw,
    }
    if parallel_method_tags is not None:
        config_extra["parallel_method_tags"] = parallel_method_tags
    run_dir = utils.setupRunDirectory(
        dataset,
        results_base_dir=results_dir,
        lm_name=role_model_names["default"],
        methods=methods,
        config_extra=config_extra,
    )

    # 5. 获取语言模型配置路径
    config_lm_path = utils.getLmConfigPath(os.path.dirname(__file__))

    # 6. 运行实验
    spent = 0.0
    prompter = MultiHopPrompter()
    parser = MultiHopParser()

    pg = parallel_got or {
        "got_hops": 4,
        "got_branch_k": 2,
        "got_critic_retries": 3,
    }

    if pw > 1:
        assert parallel_method_tags is not None
        if vis_store is not None:
            logging.warning("进程池并行模式下已忽略 realtime_vis（子进程不向主进程推送事件）。")

        tasks: List[Dict[str, Any]] = []
        for item in selected:
            tasks.append(
                {
                    "item": item,
                    "run_dir": run_dir,
                    "config_lm_path": config_lm_path,
                    "role_model_names": dict(role_model_names),
                    "method_tags": parallel_method_tags,
                    "got_hops": int(pg.get("got_hops", 4)),
                    "got_branch_k": int(pg.get("got_branch_k", 2)),
                    "got_critic_retries": int(pg.get("got_critic_retries", 3)),
                }
            )

        total = len(tasks)
        done = 0
        with ProcessPoolExecutor(max_workers=pw) as ex:
            futures = {ex.submit(_multi_hop_pool_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                done += 1
                res = fut.result()
                c = float(res.get("cost") or 0.0)
                spent += c
                budget -= c
                sid = res.get("_id", "")
                chain = "-".join(parallel_method_tags)
                if res.get("ok"):
                    print(
                        f"  [并行 {done}/{total}] 完成 _id={sid} "
                        f"串行[{chain}] 合计 cost=${c:.4f}"
                    )
                else:
                    print(f"  [并行 {done}/{total}] 失败 _id={sid} err={res.get('error', '')[:200]}")
                if budget <= 0:
                    logging.error("预算耗尽；已提交的任务仍会由子进程跑完，请提高 budget 或减少样本。")
        return spent

    for idx, item in enumerate(selected, start=1):
        print(f"正在运行第 {idx}/{len(selected)} 个问题，_id={item.get('_id', '')}")
        if budget <= 0:
            logging.error("Budget depleted, stopping.")
            break

        for method in methods:
            print(f"  方法: {method.__name__}")
            if budget <= 0:
                break

            lm = _make_lm_for_method(method, role_model_names, config_lm_path)

            # 执行单个方法
            run_id = f"{method.__name__}:{item.get('_id', idx)}"
            print(f"  run_id: {run_id}")
            if vis_store is not None:
                vis_store.publish(run_id, {
                    "type": "run_meta",
                    "run_id": run_id,
                    "sample_id": str(item.get("_id", "")),
                    "method": method.__name__,
                    "question": item.get("question", ""),
                })

            def _event_sink(payload, _rid=run_id):
                if vis_store is not None:
                    vis_store.publish(_rid, payload)

            cost = utils.runSingleMethod(item=item, method=method, lm=lm, prompter=prompter, parser=parser,
                                         run_dir=run_dir, event_sink=_event_sink if vis_store is not None else None)

            budget -= cost
            spent += cost

    return spent


if __name__ == "__main__":
    # 默认入口：未指定 num_samples / sample_id 时跑全量；指定 num_samples 则随机抽样
    # 支持的数据集：hotpotqa, musique_ans, musique_full
    parser = argparse.ArgumentParser(description="多跳问答 GoT 实验")
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                        choices=["hotpotqa", "musique_ans", "musique_full"],
                        help="数据集名称")
    parser.add_argument("--budget", type=float, default=5.0,
                        help="预算（美元）")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        metavar="N",
        help="随机抽取 N 道题；不设且未指定 --sample_id 时跑完整数据集",
    )
    parser.add_argument("--realtime_vis", action="store_true",
                        help="开启实时推理图可视化服务")
    parser.add_argument("--vis_host", type=str, default="127.0.0.1",
                        help="实时可视化服务地址")
    parser.add_argument("--vis_port", type=int, default=8765,
                        help="实时可视化服务端口")
    parser.add_argument(
        "--sample_id",
        type=str,
        default="",
        help="只跑该 id/_id 对应的题目（优先于 --num_samples 与全量）",
    )
    parser.add_argument(
        "--got_hops",
        type=int,
        default=4,
        help="multiAgentGoT 图中 hop 槽位（应 ≥ 数据集中该集最大跳数；Hotpot 多为 2，MuSiQue 常 2–4）",
    )
    parser.add_argument(
        "--got_branch_k",
        type=int,
        default=2,
        help="每跳 Retriever/Reasoner 并行分支数；设为 1 可明显缩短时间（略降多样性）",
    )
    parser.add_argument(
        "--got_critic_retries",
        type=int,
        default=3,
        help="Critic REJECT 后每跳最大回溯次数；减小可加速（可能更易早停在不理想证据上）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行进程数（1=整批串行）。>1 时每题一个进程并行，题内 methods（如 cot→tot→got→multiAgentGoT）仍串行",
    )
    args = parser.parse_args()

    # 数据集路径和大小映射
    DATASET_CONFIG = {
        "hotpotqa": {
            "path": os.path.join(os.path.dirname(__file__), "..", "dataset", "hotpotQA",
                                 "hotpot_dev_distractor_v1.json"),
            "size": 7405,
        },
        "musique_ans": {
            "path": os.path.join(os.path.dirname(__file__), "..", "dataset", "MuSiQue", "musique_ans_v1.0_dev.jsonl"),
            "size": 2417,
        },
        "musique_full": {
            "path": os.path.join(os.path.dirname(__file__), "..", "dataset", "MuSiQue", "musique_full_v1.0_dev.jsonl"),
            "size": 4834,
        },
    }

    config = DATASET_CONFIG[args.dataset]
    data_path = config["path"]
    len_data = config["size"]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件不存在: {data_path}")

    seed = int(time.time())
    # seed = 42
    random.seed(seed)

    if args.sample_id and args.sample_id.strip():
        target = args.sample_id.strip()
        data = utils.loadMultiHopData(data_path, max_samples=len_data)
        hits = []
        for i, item in enumerate(data):
            sid = str(item.get("_id") or item.get("id") or "")
            if sid == target:
                hits.append(i)
        if not hits:
            raise ValueError(f"未在数据集中找到 sample_id={target}")
        samples = hits
    elif args.num_samples is not None:
        if args.num_samples < 1:
            raise ValueError("--num_samples 须为正整数")
        samples = random.sample(range(len_data), min(args.num_samples, len_data))
    else:
        samples = list(range(len_data))


    def multiAgentGoT_configured(max_subquestions: int = 4) -> operations.GraphOfOperations:
        nh = max(1, int(max_subquestions or 1), int(args.got_hops))
        return multiAgentGoT(
            num_branches=nh,
            local_branch_k=int(args.got_branch_k),
            critic_max_retries=max(1, int(args.got_critic_retries)),
        )


    approaches = [cot, tot, got, multiAgentGoT_configured]

    # 角色模型分配（方案A：一个角色一个智能体/模型实例）
    # 角色模型分配
    # - "__lite__": 轻量模型轮换池（见 graph_of_thoughts.language_models.rotating）
    # - "__heavy__": 复杂模型轮换池（见 graph_of_thoughts.language_models.rotating）
    # - 其它字符串：单一模型（config.json 的 key）
    role_model_names = {
        "planner": "__heavy__",
        "retriever": "__lite__",
        "reasoner": "__heavy__",
        "critic": "__heavy__",
        "default": "__lite__",
    }

    print(f"数据集: {args.dataset}")
    print(f"语言模型: {role_model_names}")
    if args.sample_id and args.sample_id.strip():
        print(f"抽样: 指定 sample_id={args.sample_id.strip()!r}，共 {len(samples)} 条索引")
    elif args.num_samples is not None:
        print(f"抽样: 随机 {len(samples)} / {len_data}（seed={seed}）")
    else:
        print(f"抽样: 全量 {len(samples)} 题")
    print(f"样本数: {len(samples)}")
    print(f"预算: ${args.budget}")
    print(f"并行进程数: {max(1, args.workers)}")

    vis_store = None
    if args.realtime_vis and max(1, args.workers) == 1:
        vis_store = EventStore()
        start_realtime_server(vis_store, host=args.vis_host, port=args.vis_port)
        print(f"实时可视化服务已启动: http://{args.vis_host}:{args.vis_port}/")
        print("运行中的每个样本 run_id 形如: multiAgentGoT:<sample_id>")
    elif args.realtime_vis and args.workers > 1:
        print("已跳过 realtime_vis（与 --workers>1 不兼容）")

    spent = run(
        samples,
        approaches,
        args.budget,
        role_model_names,
        args.dataset,
        data_path=data_path,
        max_samples=len_data,
        vis_store=vis_store,
        parallel_workers=max(1, args.workers),
        parallel_got={
            "got_hops": args.got_hops,
            "got_branch_k": args.got_branch_k,
            "got_critic_retries": args.got_critic_retries,
        },
    )
    logging.info("Spent %s out of %s budget.", spent, args.budget)
