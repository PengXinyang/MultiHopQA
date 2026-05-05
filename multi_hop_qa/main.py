# 多跳推理示例：在 GoT 框架下对多文档问答进行 IO / CoT / ToT / GoT 多种推理方式。

import argparse
import logging
import math
import os
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils
from graph_of_thoughts import operations
from graph_of_thoughts.visualization import EventStore, start_realtime_server
from multi_hop_qa.data.multi_hop_graphs import multiAgentGoT, cot, tot, got
from multi_hop_parallel import make_lm_for_method, method_to_parallel_tag, run_parallel_methods
from multi_hop_qa.data.multi_hop_parser import MultiHopParser
from multi_hop_qa.data.multi_hop_prompter import MultiHopPrompter


# --- Run ---


def run_serial_methods(
        selected: List[Dict[str, Any]],
        methods: List[Callable[..., operations.GraphOfOperations]],
        budget: float,
        role_model_names: dict[str, str],
        config_lm_path: str,
        run_dir: str,
        vis_store: Optional[EventStore] = None,
) -> float:
    """串行运行样本与方法；保留实时可视化事件推送。"""
    spent = 0.0
    prompter = MultiHopPrompter()
    parser = MultiHopParser()

    for idx, item in enumerate(selected, start=1):
        print(f"正在运行第 {idx}/{len(selected)} 个问题，_id={item.get('_id', '')}")
        if math.isfinite(budget) and budget <= 0:
            logging.error("Budget depleted, stopping.")
            break

        for method in methods:
            print(f"  方法: {method.__name__}")
            if math.isfinite(budget) and budget <= 0:
                break

            lm = make_lm_for_method(method, role_model_names, config_lm_path)
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

            cost = utils.runSingleMethod(
                item=item,
                method=method,
                lm=lm,
                prompter=prompter,
                parser=parser,
                run_dir=run_dir,
                event_sink=_event_sink if vis_store is not None else None,
            )
            budget -= cost
            spent += cost

        n_chk = int(utils.DATASET_AGGREGATE_CHECKPOINT_EVERY)
        if n_chk > 0 and idx % n_chk == 0:
            utils.finalize_run_aggregate(
                run_dir,
                progress_completed_n=idx,
                print_table=False,
            )

    utils.finalize_run_aggregate(run_dir)
    return spent


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
        run_label: str = "",
        experiment_config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    加载多跳问答数据集（HotpotQA / MuSiQue），对指定样本和指定方法运行 GoT 框架，
    并将每次运行的 GRS（Graph Reasoning State）输出到 `results/` 目录下。
    
    Args:
        data_ids: 要运行的样本索引列表，None 或空列表表示全部
        methods: 方法列表，每个方法返回一个 GraphOfOperations
        budget: 预算（美元）；非有限正数表示无上限；有限正数时串行模式下用尽则停止后续题
        role_model_names: 语言模型名称
        dataset: 运行哪个数据集
        data_path: 数据集路径，None 则使用默认 HotpotQA
        max_samples: 最大加载样本数
        parallel_workers: >1 时按「每题一个进程」并行；同一题内 methods 顺序串行执行
    
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

    return run_selected_items(
        selected=selected,
        methods=methods,
        budget=budget,
        role_model_names=role_model_names,
        dataset=dataset,
        data_path=data_path,
        max_samples=max_samples,
        data_ids=data_ids[:len(selected)],
        vis_store=vis_store,
        parallel_workers=parallel_workers,
        run_label=run_label,
        experiment_config=experiment_config,
    )


def run_selected_items(
        selected: List[Dict[str, Any]],
        methods: List[Callable[..., operations.GraphOfOperations]],
        budget: float,
        role_model_names: dict[str, str],
        dataset: str,
        data_path: Any = None,
        max_samples: int = 100,
        data_ids: Any = None,
        vis_store: Optional[EventStore] = None,
        parallel_workers: int = 1,
        run_label: str = "",
        experiment_config: Optional[Dict[str, Any]] = None,
) -> float:
    """Run already-selected items. Used by mixed-dataset experiments and normal runs."""

    if not math.isfinite(budget) or budget < 0:
        budget = float("inf")

    pw = max(1, int(parallel_workers or 1))
    parallel_method_tags: Optional[List[str]] = None
    if pw > 1:
        try:
            parallel_method_tags = [method_to_parallel_tag(m) for m in methods]
        except ValueError as e:
            raise ValueError(
                "parallel_workers>1 时，methods 须为可序列化标签："
                "io / cot / tot / got，以及任意名称以 multiAgentGoT 开头的工厂。"
            ) from e

    # 4. 创建运行目录和配置
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    config_extra: Dict[str, Any] = {
        "data_path": data_path,
        "data_ids": data_ids,
        "budget": budget if math.isfinite(budget) else None,
        "budget_unlimited": not math.isfinite(budget),
        "max_samples": max_samples,
        "parallel_workers": pw,
    }
    if parallel_method_tags is not None:
        config_extra["parallel_method_tags"] = parallel_method_tags
    if run_label:
        config_extra["run_label"] = run_label
    if experiment_config:
        config_extra.update(experiment_config)

    lm_name = role_model_names["default"]
    if run_label:
        lm_name = f"{lm_name}_{sanitize_run_label(run_label)}"
    run_dir = utils.setupRunDirectory(
        dataset,
        results_base_dir=results_dir,
        lm_name=lm_name,
        methods=methods,
        config_extra=config_extra,
    )

    # 5. 获取语言模型配置路径
    config_lm_path = utils.getLmConfigPath(os.path.dirname(__file__))

    # 6. 运行实验
    if pw > 1:
        assert parallel_method_tags is not None
        if vis_store is not None:
            logging.warning("进程池并行模式下已忽略 realtime_vis（子进程不向主进程推送事件）。")
        return run_parallel_methods(
            selected=selected,
            run_dir=run_dir,
            config_lm_path=config_lm_path,
            role_model_names=role_model_names,
            method_tags=parallel_method_tags,
            parallel_workers=pw,
            budget=budget,
        )

    return run_serial_methods(
        selected=selected,
        methods=methods,
        budget=budget,
        role_model_names=role_model_names,
        config_lm_path=config_lm_path,
        run_dir=run_dir,
        vis_store=vis_store,
    )


def sanitize_run_label(label: str) -> str:
    """Convert an experiment label into a filesystem-friendly suffix."""
    keep = []
    for ch in str(label or "").strip():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多跳问答 GoT 实验")
    parser.add_argument("--dataset", type=str, default="mixed",
                        choices=["mixed", "hotpotqa", "musique_ans", "musique_full"],
                        help="数据集名称；默认 mixed 表示从 HotpotQA 和 MuSiQue 各抽 --num_samples 条")
    parser.add_argument(
        "--budget",
        type=float,
        default=float("inf"),
        help="预算上限（美元）；默认无上限。设为有限正数时，串行模式用尽后不再跑后续样本；并行模式仅记录日志。负数或非数视为无上限",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        metavar="N",
        help="随机抽取 N 道题；不设且未指定 --sample_id 时跑完整数据集",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机抽样种子；不指定时使用当前时间。例如 --seed 42",
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
        "--workers",
        type=int,
        default=1,
        help="并行进程数（1=整批串行）。>1 时每题一个进程并行，题内按 approaches 列表顺序串行",
    )
    parser.add_argument(
        "--aggregate_only",
        type=str,
        default="",
        metavar="RUN_DIR",
        help="不跑实验：仅扫描 *.summary.json，生成 dataset_aggregate_metrics.json、dataset_aggregate_table.json 并打印表后退出",
    )
    return parser


def handle_aggregate_only(aggregate_only: str) -> bool:
    agg_path = (aggregate_only or "").strip()
    if not agg_path:
        return False

    run_dir_abs = os.path.abspath(agg_path)
    if not os.path.isdir(run_dir_abs):
        print(f"目录不存在: {run_dir_abs}", file=sys.stderr)
        sys.exit(1)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    utils.finalize_run_aggregate(run_dir_abs)
    return True


def get_dataset_config() -> Dict[str, Dict[str, Any]]:
    return {
        "mixed": {
            "path": None,
            "size": None,
        },
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


def select_sample_indices(args: argparse.Namespace, data_path: str, len_data: int) -> List[int]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件不存在: {data_path}")

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
        return hits
    elif args.num_samples is not None:
        if args.num_samples < 1:
            raise ValueError("--num_samples 须为正整数")
        return random.sample(range(len_data), min(args.num_samples, len_data))
    return list(range(len_data))


def load_mixed_selected_items(args: argparse.Namespace, configs: Dict[str, Dict[str, Any]]) -> tuple:
    """
    Default small benchmark: sample N HotpotQA items and N MuSiQue items, then merge them.

    Returns (selected_items, mixed_data_ids, mixed_data_paths, max_samples_loaded).
    """
    if args.sample_id and args.sample_id.strip():
        raise ValueError("--dataset mixed 不支持 --sample_id；请指定 --dataset hotpotqa 或 musique_ans。")
    if args.num_samples is None:
        raise ValueError("--dataset mixed 需要显式指定 --num_samples，表示每个数据集抽取多少条。")
    if args.num_samples < 1:
        raise ValueError("--num_samples 须为正整数")

    selected: List[Dict[str, Any]] = []
    data_ids: Dict[str, List[int]] = {}
    data_paths: Dict[str, str] = {}
    for dataset_name in ("hotpotqa", "musique_ans"):
        cfg = configs[dataset_name]
        data_path = cfg["path"]
        len_data = int(cfg["size"])
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在: {data_path}")
        ids = random.sample(range(len_data), min(args.num_samples, len_data))
        data = utils.loadMultiHopData(data_path, max_samples=len_data)
        for idx in ids:
            item = dict(data[idx])
            item["_source_dataset"] = dataset_name
            selected.append(item)
        data_ids[dataset_name] = ids
        data_paths[dataset_name] = data_path
    return selected, data_ids, data_paths, sum(int(configs[name]["size"]) for name in ("hotpotqa", "musique_ans"))


def default_role_model_names() -> Dict[str, str]:
    # 角色模型分配（方案A：一个角色一个智能体/模型实例）
    # - "__lite__": 轻量模型轮换池（见 graph_of_thoughts.language_models.rotating）
    # - "__heavy__": 复杂模型轮换池（见 graph_of_thoughts.language_models.rotating）
    # - 其它字符串：单一模型（config.json 的 key）
    return {
        "planner": "__heavy__",
        "retriever": "__lite__",
        "reasoner": "__heavy__",
        "critic": "__heavy__",
        "default": "__lite__",
    }


def print_run_config(
        args: argparse.Namespace,
        role_model_names: Dict[str, str],
        samples: Any,
        len_data: int,
        seed: int,
) -> None:
    print(f"数据集: {args.dataset}")
    print(f"语言模型: {role_model_names}")
    if args.dataset == "mixed" and isinstance(samples, dict):
        total_selected = sum(len(v) for v in samples.values())
        detail = ", ".join(f"{k}={len(v)}" for k, v in samples.items())
        print(f"抽样: 混合数据集 {detail}，共 {total_selected} 条（seed={seed}）")
    elif args.sample_id and args.sample_id.strip():
        print(f"抽样: 指定 sample_id={args.sample_id.strip()!r}，共 {len(samples)} 条索引")
    elif args.num_samples is not None:
        print(f"抽样: 随机 {len(samples)} / {len_data}（seed={seed}）")
    else:
        print(f"抽样: 全量 {len(samples)} 题")
    sample_count = sum(len(v) for v in samples.values()) if isinstance(samples, dict) else len(samples)
    print(f"样本数: {sample_count}")
    print(
        f"预算: {'无上限' if not math.isfinite(args.budget) or args.budget < 0 else f'${args.budget}'}"
    )
    print(f"并行进程数: {max(1, args.workers)}")


def start_realtime_vis(args: argparse.Namespace) -> Optional[EventStore]:
    vis_store = None
    if args.realtime_vis and max(1, args.workers) == 1:
        vis_store = EventStore()
        start_realtime_server(vis_store, host=args.vis_host, port=args.vis_port)
        print(f"实时可视化服务已启动: http://{args.vis_host}:{args.vis_port}/")
        print("运行中的每个样本 run_id 形如: multiAgentGoT:<sample_id>")
    elif args.realtime_vis and args.workers > 1:
        print("已跳过 realtime_vis（与 --workers>1 不兼容）")
    return vis_store


def main() -> None:
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    # 默认入口：未指定 num_samples / sample_id 时跑全量；指定 num_samples 则随机抽样
    # 支持的数据集：hotpotqa, musique_ans, musique_full
    parser = build_arg_parser()
    args = parser.parse_args()

    if handle_aggregate_only(args.aggregate_only):
        sys.exit(0)

    seed = int(args.seed) if args.seed is not None else int(time.time())
    random.seed(seed)

    configs = get_dataset_config()
    config = configs[args.dataset]
    if args.dataset == "mixed":
        selected_items, samples, data_path, len_data = load_mixed_selected_items(args, configs)
    else:
        data_path = config["path"]
        len_data = config["size"]
        samples = select_sample_indices(args, data_path, len_data)
        selected_items = None

    approaches = [multiAgentGoT, cot, tot, got]
    # approaches = [multiAgentGoT]
    role_model_names = default_role_model_names()

    print_run_config(args, role_model_names, samples, len_data, seed)
    vis_store = start_realtime_vis(args)

    _bud = args.budget
    if args.dataset == "mixed":
        spent = run_selected_items(
            selected=selected_items,
            methods=approaches,
            budget=_bud,
            role_model_names=role_model_names,
            dataset=args.dataset,
            data_path=data_path,
            max_samples=len_data,
            data_ids=samples,
            vis_store=vis_store,
            parallel_workers=max(1, args.workers),
            experiment_config={
                "mixed_dataset_components": ["hotpotqa", "musique_ans"],
                "samples_per_dataset": args.num_samples,
                "sample_seed": seed,
            },
        )
    else:
        spent = run(
            samples,
            approaches,
            _bud,
            role_model_names,
            args.dataset,
            data_path=data_path,
            max_samples=len_data,
            vis_store=vis_store,
            parallel_workers=max(1, args.workers),
        )
    if math.isfinite(_bud) and _bud >= 0:
        logging.info("Spent %s out of %s budget.", spent, _bud)
    else:
        logging.info("Spent %s (无预算上限).", spent)

    if args.realtime_vis and args.workers <= 1:
        logging.info("等待前端拉取最终可视化事件...")
        time.sleep(3)


if __name__ == "__main__":
    main()
