# 消融实验 A1: w/o 动态回溯
# 验证者仅评分但不触发回溯纠错（critic_max_retries=0）

import argparse
import logging
import math
import os
import random
import sys
import time
from functools import partial
from typing import Callable, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph_of_thoughts import operations
from multi_hop_qa.data.multi_hop_graphs import multiAgentGoT
from multi_hop_qa.main import (
    default_role_model_names,
    get_dataset_config,
    handle_aggregate_only,
    load_mixed_selected_items,
    print_run_config,
    run,
    run_selected_items,
    select_sample_indices,
    start_realtime_vis,
)


def backtrack_ablation_variants(
    name: str,
) -> List[tuple[str, List[Callable[..., operations.GraphOfOperations]], Dict[str, str]]]:
    """
    消融实验 A1：对比不同 critic_max_retries 值下的性能与成本。

    - no_backtrack: critic_max_retries=0，验证者仅评分不回溯
    - full (baseline): critic_max_retries=3，完整 GoT-MAS 默认配置
    """
    role_names = default_role_model_names()
    no_bt = partial(multiAgentGoT, critic_max_retries=0)
    no_bt.__name__ = "multiAgentGoT"
    variants = {
        "no_backtrack": (
            [no_bt],
            role_names,
        ),
        "full": (
            [multiAgentGoT],
            role_names,
        ),
    }
    if name == "all":
        order = ["no_backtrack", "full"]
        return [(key, *variants[key]) for key in order]
    methods, role_model = variants[name]
    return [(name, methods, role_model)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="消融实验 A1: w/o 动态回溯 (critic_max_retries=0 vs 默认配置)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mixed",
        choices=["mixed", "hotpotqa", "musique_ans", "musique_full"],
        help="数据集名称；默认 mixed 表示从 HotpotQA 和 MuSiQue 各抽 --num_samples 条",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=float("inf"),
        help="预算上限（美元）；默认无上限",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        metavar="N",
        help="每个数据集随机抽取 N 道题；不设时跑完整数据集",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机抽样种子；消融实验默认固定为 42",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["no_backtrack", "full", "all"],
        help="no_backtrack=无回溯, full=完整模型(对照), all=依次运行两组",
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default="",
        help="只跑该 id/_id 对应的题目（优先于 --num_samples）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行进程数（1=串行）",
    )
    parser.add_argument(
        "--realtime_vis",
        action="store_true",
        help="开启实时推理图可视化服务",
    )
    parser.add_argument("--vis_host", type=str, default="127.0.0.1")
    parser.add_argument("--vis_port", type=int, default=8765)
    parser.add_argument(
        "--aggregate_only",
        type=str,
        default="",
        metavar="RUN_DIR",
        help="不跑实验：仅扫描已有结果并生成汇总表后退出",
    )
    return parser


def main() -> None:
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    parser = build_arg_parser()
    args = parser.parse_args()

    if handle_aggregate_only(args.aggregate_only):
        sys.exit(0)

    configs = get_dataset_config()
    config = configs[args.dataset]

    seed = int(args.seed)
    random.seed(seed)
    if args.dataset == "mixed":
        selected_items, samples, data_path, len_data = load_mixed_selected_items(args, configs)
    else:
        data_path = config["path"]
        len_data = config["size"]
        samples = select_sample_indices(args, data_path, len_data)
        selected_items = None

    variants = backtrack_ablation_variants(args.variant)
    _, _, first_role_model_names = variants[0]
    print_run_config(args, first_role_model_names, samples, len_data, seed)
    print(f"回溯消融变体: {', '.join(v[0] for v in variants)}")

    vis_store = start_realtime_vis(args) if len(variants) == 1 else None
    if args.realtime_vis and len(variants) > 1:
        print("已跳过 realtime_vis（多变体运行时不推送可视化事件）")

    original_budget = args.budget
    remaining_budget = (
        float("inf")
        if not math.isfinite(original_budget) or original_budget < 0
        else original_budget
    )
    spent = 0.0

    for variant_name, methods, role_model_names in variants:
        if math.isfinite(remaining_budget) and remaining_budget <= 0:
            logging.error("预算耗尽；跳过剩余消融变体。")
            break

        print("\n" + "=" * 80)
        print(f"开始消融变体: {variant_name}")
        print(
            f"方法: {[getattr(m, '__name__', None) or getattr(getattr(m, 'func', None), '__name__', '?') for m in methods]}"
        )
        print(f"critic_max_retries: {'0 (无回溯)' if variant_name == 'no_backtrack' else '3 (默认)'}")
        print(f"角色模型: {role_model_names}")
        print("=" * 80)

        experiment_config = {
            "experiment_type": "backtrack_ablation",
            "backtrack_variant": variant_name,
            "critic_max_retries": 0 if variant_name == "no_backtrack" else 3,
            "sample_seed": seed,
        }
        if args.dataset == "mixed":
            experiment_config.update(
                {
                    "mixed_dataset_components": ["hotpotqa", "musique_ans"],
                    "samples_per_dataset": args.num_samples,
                }
            )
            variant_spent = run_selected_items(
                selected=selected_items,
                methods=methods,
                budget=remaining_budget,
                role_model_names=role_model_names,
                dataset=args.dataset,
                data_path=data_path,
                max_samples=len_data,
                data_ids=samples,
                vis_store=vis_store,
                parallel_workers=max(1, args.workers),
                run_label=f"backtrack_ablation_{variant_name}",
                experiment_config=experiment_config,
            )
        else:
            variant_spent = run(
                samples,
                methods,
                remaining_budget,
                role_model_names,
                args.dataset,
                data_path=data_path,
                max_samples=len_data,
                vis_store=vis_store,
                parallel_workers=max(1, args.workers),
                run_label=f"backtrack_ablation_{variant_name}",
                experiment_config=experiment_config,
            )
        spent += variant_spent
        if math.isfinite(remaining_budget):
            remaining_budget -= variant_spent

    print("\n" + "=" * 80)
    print(f"消融实验 A1 完成。总花费: ${spent:.4f}")
    if math.isfinite(original_budget) and original_budget >= 0:
        print(f"预算使用: ${spent:.4f} / ${original_budget:.4f}")
    print("=" * 80)

    if args.realtime_vis and args.workers <= 1 and len(variants) == 1:
        logging.info("等待前端拉取最终可视化事件...")
        time.sleep(3)


if __name__ == "__main__":
    main()
