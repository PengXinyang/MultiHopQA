# 消融实验 A3: w/o 分支探索
# 通过调整 local_branch_k 验证局部并行候选与剪枝竞争的价值
#   branch_k1: local_branch_k=1，每跳仅 1 个候选，无剪枝 → 预期正确率下降
#   branch_k3: local_branch_k=3，每跳 3 个候选 → 预期成本增加，正确率基本持平

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

BRANCH_K_LABELS = {
    "branch_k1": 1,
    "branch_k3": 3,
    "full": 2,  # 默认 local_branch_k=2
}


def branch_ablation_variants(
    name: str,
) -> List[tuple[str, List[Callable[..., operations.GraphOfOperations]], Dict[str, str]]]:
    """
    消融实验 A3：对比不同 local_branch_k 下的性能与成本。

    - branch_k1: local_branch_k=1，每跳只生成 1 个候选，完全无剪枝竞争
    - branch_k3: local_branch_k=3，每跳生成 3 个候选，探索更多但成本更高
    - full (baseline): local_branch_k=2，完整 GoT-MAS 默认配置
    """
    role_names = default_role_model_names()
    k1 = partial(multiAgentGoT, local_branch_k=1)
    k1.__name__ = "multiAgentGoT"
    k3 = partial(multiAgentGoT, local_branch_k=3)
    k3.__name__ = "multiAgentGoT"
    variants = {
        "branch_k1": (
            [k1],
            role_names,
        ),
        "branch_k3": (
            [k3],
            role_names,
        ),
        "full": (
            [multiAgentGoT],
            role_names,
        ),
    }
    if name == "all":
        order = ["branch_k1", "branch_k3", "full"]
        return [(key, *variants[key]) for key in order]
    methods, role_model = variants[name]
    return [(name, methods, role_model)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="消融实验 A3: w/o 分支探索 (local_branch_k=1/3 vs 默认 k=2)"
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
        choices=["branch_k1", "branch_k3", "full", "all"],
        help="branch_k1=每跳1候选, branch_k3=每跳3候选, full=默认k=2(对照), all=依次全部运行",
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

    variants = branch_ablation_variants(args.variant)
    _, _, first_role_model_names = variants[0]
    print_run_config(args, first_role_model_names, samples, len_data, seed)
    print(f"分支探索消融变体: {', '.join(v[0] for v in variants)}")

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

        k_value = BRANCH_K_LABELS[variant_name]
        print("\n" + "=" * 80)
        print(f"开始消融变体: {variant_name}")
        print(f"方法: {[getattr(m, '__name__', m.func.__name__) for m in methods]}")
        print(f"local_branch_k: {k_value}" + (" (无剪枝)" if k_value == 1 else f" ({'默认' if k_value == 2 else '扩展探索'})"))
        print(f"角色模型: {role_model_names}")
        print("=" * 80)

        experiment_config = {
            "experiment_type": "branch_ablation",
            "branch_variant": variant_name,
            "local_branch_k": k_value,
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
                run_label=f"branch_ablation_{variant_name}",
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
                run_label=f"branch_ablation_{variant_name}",
                experiment_config=experiment_config,
            )
        spent += variant_spent
        if math.isfinite(remaining_budget):
            remaining_budget -= variant_spent

    print("\n" + "=" * 80)
    print(f"消融实验 A3 完成。总花费: ${spent:.4f}")
    if math.isfinite(original_budget) and original_budget >= 0:
        print(f"预算使用: ${spent:.4f} / ${original_budget:.4f}")
    print("=" * 80)

    if args.realtime_vis and args.workers <= 1 and len(variants) == 1:
        logging.info("等待前端拉取最终可视化事件...")
        time.sleep(3)


if __name__ == "__main__":
    main()
