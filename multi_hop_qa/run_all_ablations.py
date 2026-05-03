# 一键运行全部消融实验（A1 / A2 / A3 + 完整模型）并生成跨实验汇总表。
#
# 实验矩阵（6 组变体）:
#   A1  w/o 动态回溯    critic_max_retries=0
#   A2a w/o 异构路由     全部用 lite 模型
#   A2b w/o 异构路由     全部用 heavy 模型
#   A3a w/o 分支探索     local_branch_k=1
#   A3b 分支扩展(对照)   local_branch_k=3
#   A0  完整 GoT-MAS     默认配置（baseline）
#
# 用法:
#   python multi_hop_qa/run_all_ablations.py --num_samples 30 --workers 4
#   python multi_hop_qa/run_all_ablations.py --dataset hotpotqa --num_samples 50 --workers 2

import argparse
import datetime
import json
import logging
import math
import os
import random
import sys
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph_of_thoughts import operations
from multi_hop_qa.data.multi_hop_graphs import multiAgentGoT
from multi_hop_qa.main import (
    default_role_model_names,
    get_dataset_config,
    load_mixed_selected_items,
    print_run_config,
    run,
    run_selected_items,
    select_sample_indices,
)
from multi_hop_qa.role_ablation import same_model_role_names
from multi_hop_qa import utils

# parallel 子进程通过此全局字典重建 multiAgentGoT 参数
import multi_hop_qa.multi_hop_parallel as _parallel_mod


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def _named_partial(func, **kwargs):
    """functools.partial with __name__ preserved (setupRunDirectory needs it)."""
    p = partial(func, **kwargs)
    p.__name__ = func.__name__
    return p


def build_all_variants() -> List[Tuple[str, str, List[Callable], Dict[str, str], Dict[str, int]]]:
    """
    Returns list of (label, display_name, methods, role_model_names, magot_overrides).

    magot_overrides: keys matching MAGOT_DEFAULTS to patch for parallel workers.
    """
    default_roles = default_role_model_names()
    return [
        (
            "A1_no_backtrack",
            "A1 w/o 动态回溯",
            [_named_partial(multiAgentGoT, critic_max_retries=0)],
            default_roles,
            {"got_critic_retries": 0},
        ),
        (
            "A2_same_lite",
            "A2 全lite模型",
            [multiAgentGoT],
            same_model_role_names("__lite__"),
            {},
        ),
        (
            "A2_same_heavy",
            "A2 全heavy模型",
            [multiAgentGoT],
            same_model_role_names("__heavy__"),
            {},
        ),
        (
            "A3_branch_k1",
            "A3 分支k=1",
            [_named_partial(multiAgentGoT, local_branch_k=1)],
            default_roles,
            {"got_branch_k": 1},
        ),
        (
            "A3_branch_k3",
            "A3 分支k=3",
            [_named_partial(multiAgentGoT, local_branch_k=3)],
            default_roles,
            {"got_branch_k": 3},
        ),
        (
            "A0_full",
            "完整 GoT-MAS",
            [multiAgentGoT],
            default_roles,
            {},
        ),
    ]


# ---------------------------------------------------------------------------
# Cross-experiment summary
# ---------------------------------------------------------------------------

def collect_cross_experiment_summary(
    completed: List[Tuple[str, str, str, float]],
) -> List[Dict[str, Any]]:
    """
    Collect aggregate metrics from each variant's run_dir.

    Args:
        completed: list of (label, display_name, run_dir, cost)
    Returns:
        list of row dicts for the summary table.
    """
    rows = []
    for label, display_name, run_dir, cost in completed:
        row: Dict[str, Any] = {
            "label": label,
            "name": display_name,
            "run_dir": run_dir,
            "cost_reported": cost,
        }
        metrics_path = os.path.join(run_dir, "dataset_aggregate_metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                agg = json.load(f)
            methods = agg.get("methods") or {}
            if methods:
                m = next(iter(methods.values()))
                row["N"] = int(m.get("num_summaries", 0))
                row["correct_rate"] = float(m.get("correct_rate", 0.0))
                row["EM_accuracy"] = float(m.get("EM_accuracy", 0.0))
                row["mean_F1"] = float(m.get("mean_F1", 0.0))
                row["total_cost"] = float(m.get("total_cost_usd", 0.0))
                row["mean_cost"] = row["total_cost"] / row["N"] if row["N"] else 0.0
        rows.append(row)
    return rows


def print_cross_experiment_table(rows: List[Dict[str, Any]]) -> None:
    """Pretty-print the ablation summary table."""
    print("\n" + "=" * 110)
    print(" " * 20 + "消 融 实 验 汇 总 表")
    print("=" * 110)
    hdr = (
        f"{'编号':<18} {'消融条件':<18} {'N':>4} "
        f"{'正确率':>8} {'EM率':>8} {'平均F1':>8} "
        f"{'总费用$':>10} {'题均费用$':>10}"
    )
    print(hdr)
    print("-" * 110)
    for r in rows:
        n = r.get("N", "—")
        cr = f"{r['correct_rate']:.4f}" if "correct_rate" in r else "—"
        em = f"{r['EM_accuracy']:.4f}" if "EM_accuracy" in r else "—"
        f1 = f"{r['mean_F1']:.4f}" if "mean_F1" in r else "—"
        tc = f"{r['total_cost']:.4f}" if "total_cost" in r else "—"
        mc = f"{r['mean_cost']:.4f}" if "mean_cost" in r else "—"
        print(
            f"{r['label']:<18} {r['name']:<18} {n:>4} "
            f"{cr:>8} {em:>8} {f1:>8} "
            f"{tc:>10} {mc:>10}"
        )
    print("=" * 110)


def save_cross_experiment_json(
    rows: List[Dict[str, Any]], output_path: str
) -> None:
    serializable = []
    for r in rows:
        sr = {k: v for k, v in r.items() if k != "run_dir"}
        sr["run_dir"] = str(r.get("run_dir", ""))
        serializable.append(sr)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.datetime.now().isoformat(),
                "variants": serializable,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n汇总 JSON 已保存: {output_path}")


# ---------------------------------------------------------------------------
# Detect new run_dir
# ---------------------------------------------------------------------------

def _list_subdirs(parent: str) -> set:
    if not os.path.isdir(parent):
        return set()
    return {
        d for d in os.listdir(parent)
        if os.path.isdir(os.path.join(parent, d))
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一键运行全部消融实验 (A0-A3) 并生成汇总表"
    )
    parser.add_argument(
        "--dataset", type=str, default="mixed",
        choices=["mixed", "hotpotqa", "musique_ans", "musique_full"],
    )
    parser.add_argument("--budget", type=float, default=float("inf"))
    parser.add_argument("--num_samples", type=int, default=None, metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="每组实验内部的并行进程数（题间并行）",
    )
    parser.add_argument(
        "--variants", type=str, default="all",
        help=(
            "要运行的变体，逗号分隔，如 'A1_no_backtrack,A0_full'；"
            "默认 all 运行全部 6 组"
        ),
    )
    parser.add_argument(
        "--sample_id", type=str, default="",
        help="只跑该 id 对应的单题（调试用）",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

    parser = build_arg_parser()
    args = parser.parse_args()

    configs = get_dataset_config()
    seed = int(args.seed)
    random.seed(seed)

    # ---- 加载数据、抽样（仅做一次，所有变体共享同一题目集） ----
    if args.dataset == "mixed":
        selected_items, samples, data_path, len_data = load_mixed_selected_items(args, configs)
    else:
        config = configs[args.dataset]
        data_path = config["path"]
        len_data = config["size"]
        samples = select_sample_indices(args, data_path, len_data)
        selected_items = None

    # ---- 选择要运行的变体 ----
    all_variants = build_all_variants()
    if args.variants.strip().lower() == "all":
        variants = all_variants
    else:
        wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
        variants = [v for v in all_variants if v[0] in wanted]
        if not variants:
            print(f"未匹配到任何变体。可选: {[v[0] for v in all_variants]}")
            sys.exit(1)

    print_run_config(args, default_role_model_names(), samples, len_data, seed)
    print(f"计划运行 {len(variants)} 组消融变体:")
    for label, display, *_ in variants:
        print(f"  - {label}: {display}")
    print()

    # ---- 确定结果目录的父级，用于检测新建 run_dir ----
    results_base = os.path.join(os.path.dirname(__file__), "results")
    dataset_dir = os.path.join(results_base, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    original_budget = args.budget
    remaining_budget = (
        float("inf") if not math.isfinite(original_budget) or original_budget < 0
        else original_budget
    )
    total_spent = 0.0
    completed: List[Tuple[str, str, str, float]] = []

    # ---- 保存 MAGOT_DEFAULTS 原始值 ----
    original_magot = dict(_parallel_mod.MAGOT_DEFAULTS)

    for label, display_name, methods, role_model_names, magot_overrides in variants:
        if math.isfinite(remaining_budget) and remaining_budget <= 0:
            logging.error("预算耗尽；跳过剩余变体。")
            break

        # ---- 为并行 workers 注入正确的图参数 ----
        _parallel_mod.MAGOT_DEFAULTS.update(original_magot)  # reset
        if magot_overrides:
            _parallel_mod.MAGOT_DEFAULTS.update(magot_overrides)

        print("\n" + "=" * 80)
        print(f"▶ [{label}] {display_name}")
        print(f"  方法: {[getattr(m, '__name__', m.func.__name__) for m in methods]}")
        print(f"  角色模型: {role_model_names}")
        if magot_overrides:
            print(f"  并行参数覆盖: {magot_overrides}")
        print("=" * 80)

        before_dirs = _list_subdirs(dataset_dir)

        experiment_config = {
            "experiment_type": "ablation",
            "ablation_variant": label,
            "sample_seed": seed,
        }
        if magot_overrides:
            experiment_config.update(magot_overrides)

        run_label = f"ablation_{label}"

        if args.dataset == "mixed":
            if selected_items is None:
                raise RuntimeError("mixed 数据集需要 selected_items")
            experiment_config.update({
                "mixed_dataset_components": ["hotpotqa", "musique_ans"],
                "samples_per_dataset": args.num_samples,
            })
            variant_spent = run_selected_items(
                selected=selected_items,
                methods=methods,
                budget=remaining_budget,
                role_model_names=role_model_names,
                dataset=args.dataset,
                data_path=data_path,
                max_samples=len_data,
                data_ids=samples,
                vis_store=None,
                parallel_workers=max(1, args.workers),
                run_label=run_label,
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
                vis_store=None,
                parallel_workers=max(1, args.workers),
                run_label=run_label,
                experiment_config=experiment_config,
            )

        total_spent += variant_spent
        if math.isfinite(remaining_budget):
            remaining_budget -= variant_spent

        # ---- 检测本次新建的 run_dir ----
        after_dirs = _list_subdirs(dataset_dir)
        new_dirs = sorted(after_dirs - before_dirs)
        if new_dirs:
            run_dir = os.path.join(dataset_dir, new_dirs[-1])
        else:
            run_dir = "UNKNOWN"
            print(f"  ⚠ 未检测到新建的 run_dir")

        completed.append((label, display_name, run_dir, variant_spent))
        print(f"  ✓ {label} 完成，花费 ${variant_spent:.4f}，目录: {run_dir}")

    # ---- 恢复 MAGOT_DEFAULTS ----
    _parallel_mod.MAGOT_DEFAULTS.update(original_magot)

    # ---- 汇总表格 ----
    print("\n\n" + "#" * 110)
    print("#  全部消融实验完成")
    print(f"#  总花费: ${total_spent:.4f}")
    print("#" * 110)

    rows = collect_cross_experiment_summary(completed)
    print_cross_experiment_table(rows)

    summary_path = os.path.join(
        results_base,
        f"ablation_summary_{args.dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    save_cross_experiment_json(rows, summary_path)


if __name__ == "__main__":
    main()
