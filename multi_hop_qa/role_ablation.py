# 多智能体角色分工消融实验入口。

import argparse
import logging
import math
import os
import random
import sys
import time
from typing import Callable, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph_of_thoughts import operations
from multi_hop_qa.data.multi_hop_graphs import got, multiAgentGoT
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


def same_model_role_names(model_name: str) -> Dict[str, str]:
    """Use one model group for every GoT-MAS role to remove heterogeneous routing."""
    return {
        "planner": model_name,
        "retriever": model_name,
        "reasoner": model_name,
        "critic": model_name,
        "default": model_name,
    }


def role_ablation_variants(
    name: str,
) -> List[tuple[str, List[Callable[..., operations.GraphOfOperations]], Dict[str, str]]]:
    """
    Build runnable variants for thesis section 4.4.1.

    The returned tuple is (variant_name, methods, role_model_names). Running `all`
    produces separate run directories while reusing the same sampled questions.
    """
    variants = {
        "role_routed": (
            [multiAgentGoT],
            default_role_model_names(),
        ),
        "same_lite": (
            [multiAgentGoT],
            same_model_role_names("__lite__"),
        ),
        "same_heavy": (
            [multiAgentGoT],
            same_model_role_names("__heavy__"),
        ),
        "single_agent_got": (
            [got],
            {"default": "__heavy__"},
        ),
    }
    if name == "all":
        # order = ["single_agent_got", "same_lite", "same_heavy", "role_routed"]
        order = ["same_lite", "same_heavy", "role_routed"]
        return [(key, *variants[key]) for key in order]
    methods, role_names = variants[name]
    return [(name, methods, role_names)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GoT-MAS 角色分工消融实验")
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
        help="预算上限（美元）；默认无上限。负数或非数视为无上限",
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
        default=42,
        help="随机抽样种子；消融实验默认固定为 42",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=[
            "role_routed",
            "same_lite",
            "same_heavy",
            "single_agent_got",
            "all",
        ],
        help=(
            "角色分工消融变体：role_routed=默认异构角色路由，"
            "same_lite=所有角色轻模型，same_heavy=所有角色重模型，"
            "single_agent_got=单智能体 GoT，all=依次运行全部变体"
        ),
    )
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
        help="并行进程数（1=整批串行）。>1 时每题一个进程并行，题内方法串行",
    )
    parser.add_argument(
        "--realtime_vis",
        action="store_true",
        help="开启实时推理图可视化服务；仅单变体且 --workers=1 时建议使用",
    )
    parser.add_argument("--vis_host", type=str, default="127.0.0.1", help="实时可视化服务地址")
    parser.add_argument("--vis_port", type=int, default=8765, help="实时可视化服务端口")
    parser.add_argument(
        "--aggregate_only",
        type=str,
        default="",
        metavar="RUN_DIR",
        help="不跑实验：仅扫描 *.summary.json 并生成汇总表后退出",
    )
    return parser


def main() -> None:
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

    variants = role_ablation_variants(args.variant)
    _, _, first_role_model_names = variants[0]
    print_run_config(args, first_role_model_names, samples, len_data, seed)
    print("角色消融变体: " + ", ".join(v[0] for v in variants))

    vis_store = start_realtime_vis(args) if len(variants) == 1 else None
    if args.realtime_vis and len(variants) > 1:
        print("已跳过 realtime_vis（多变体运行时不推送可视化事件）")

    original_budget = args.budget
    remaining_budget = float("inf") if not math.isfinite(original_budget) or original_budget < 0 else original_budget
    spent = 0.0

    for variant_name, methods, role_model_names in variants:
        if math.isfinite(remaining_budget) and remaining_budget <= 0:
            logging.error("预算耗尽；跳过剩余角色消融变体。")
            break

        print("\n" + "=" * 80)
        print(f"开始角色消融变体: {variant_name}")
        print(f"方法: {[m.__name__ for m in methods]}")
        print(f"角色模型: {role_model_names}")
        print("=" * 80)

        experiment_config = {
            "experiment_type": "role_ablation",
            "role_ablation_variant": variant_name,
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
                run_label=f"role_ablation_{variant_name}",
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
                run_label=f"role_ablation_{variant_name}",
                experiment_config=experiment_config,
            )
        spent += variant_spent
        if math.isfinite(remaining_budget):
            remaining_budget -= variant_spent

    if math.isfinite(original_budget) and original_budget >= 0:
        logging.info("Spent %s out of %s budget.", spent, original_budget)
    else:
        logging.info("Spent %s (无预算上限).", spent)

    if args.realtime_vis and args.workers <= 1 and len(variants) == 1:
        logging.info("等待前端拉取最终可视化事件...")
        time.sleep(3)


if __name__ == "__main__":
    main()
