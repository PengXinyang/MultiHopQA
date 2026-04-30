import datetime
import glob
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import score
from data_loader import loadDataset, DatasetType
from graph_of_thoughts import controller
from graph_of_thoughts.auto_goo_designer import (
    requestGotGooDesign,
    saveDesignResult,
)
from graph_of_thoughts.goo_builder import (
    loadGooDesignFromFile,
    buildGraphFromGooDesign,
)


# --- 数据加载与上下文格式化 ---


def loadMultiHopData(path: str, max_samples: int = None) -> List[Dict]:
    """
    加载多跳问答数据集（支持 HotpotQA 和 MuSiQue），返回字典格式。
    
    此函数自动检测数据集类型并返回统一格式的字典列表，兼容旧版代码。
    如需更丰富的功能（如 MultiHopSample 对象、过滤、统计），请使用 data_loader 模块。
    """
    return loadDataset(path, DatasetType.AUTO, max_samples, return_dict=True)


def contextToText(context: List) -> str:
    """将 Hotpot 的 context（[title, [sentences]] 列表）拼接成一段可读文本。"""
    parts = []
    for item in context:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sents = item[0], item[1]
            para = " ".join(sents) if isinstance(sents, list) else str(sents)
            parts.append(f"[{title}]\n{para}")
        else:
            parts.append(str(item))
    return "\n\n".join(parts)


def contextToTextWithIndices(context: List) -> str:
    """与 context_to_text 类似，但每一句单独一行，并带上 (title, idx) 方便定位句子。"""
    parts = []
    for item in context:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sents = item[0], item[1]
            for idx, sent in enumerate(sents):
                parts.append(f"({title}, {idx}) {sent}")
        else:
            parts.append(str(item))
    return "\n".join(parts)


# --- 运行辅助函数 ---


def setupRunDirectory(
        dataset: str,
        results_base_dir: str,
        lm_name: str,
        methods: List[Callable],
        config_extra: Dict = None,
) -> str:
    """
    创建本次运行的结果目录，保存配置文件，设置日志。
    
    Args:
        dataset: 运行哪个数据集
        results_base_dir: 结果根目录（如 "results"）
        lm_name: 语言模型名称
        methods: 方法列表（用于命名和创建子目录）
        config_extra: 额外的配置信息（会合并到 config.json）
    
    Returns:
        run_dir: 本次运行的目录路径
    """
    os.makedirs(results_base_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    method_names = "-".join([m.__name__ for m in methods])
    run_dir = os.path.join(results_base_dir, dataset, f"{lm_name}_{method_names}_{timestamp}")
    os.makedirs(run_dir)

    # 为每个方法创建子目录
    for method in methods:
        os.makedirs(os.path.join(run_dir, method.__name__), exist_ok=True)

    # 保存配置
    config = {
        "lm": lm_name,
        "methods": [m.__name__ for m in methods],
        "timestamp": timestamp,
    }
    if config_extra:
        config.update(config_extra)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 设置日志
    logging.basicConfig(
        filename=os.path.join(run_dir, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    return run_dir


def buildProblemParams(item: Dict, method_name: str) -> Dict:
    """
    将数据集中的一个样本转换为 GoT 框架所需的 problem_params 字典。
    
    Args:
        item: 数据集中的一条样本（统一格式，由 data_loader 加载）
        method_name: 方法名称（如 "io", "cot", "tot", "got"）
    
    Returns:
        problem_params: GoT Controller 所需的参数字典
    """
    context = item.get("context", [])
    context_text = contextToText(context)
    num_docs = len(context) if context else 10
    num_hops = int(item.get("num_hops", 2) or 2)
    decomposition = item.get("question_decomposition", []) or []
    dynamic_branches = len(decomposition) if decomposition else num_hops
    dynamic_branches = max(1, min(8, int(dynamic_branches)))

    return {
        "question": item.get("question", ""),
        "context": context,
        "context_text": context_text,
        "num_docs": num_docs,
        "num_hops": num_hops,
        "max_subquestions": dynamic_branches,
        "answer": "",
        "current": "",
        "ground_truth_answer": item.get("answer", ""),
        "ground_truth_sp": item.get("supporting_facts", []),
        "answer_aliases": item.get("answer_aliases", []),
        "answerable": item.get("answerable", True),
        "question_decomposition": decomposition,
        "precomputed_subquestions": [d.get("question", "") for d in decomposition if d.get("question", "")],
        "phase": 0,
        "agent_role": "planner" if method_name.startswith("multiAgentGoT") else "",
        "sub_id": -1,
        "subquestion": "",
        "evidence_spans": [],
        "evidence_summary": "",
        "partial_answer": "",
        "confidence": 0.0,
        "candidate_answers": [],
        "online_reasoning_score": 0.0,
        "offline_metric_score": 0.0,
        "solve_score_threshold": 0.8,
        "line_score_threshold": 0.7,
        "low_trust_penalty": 0.35,
        "method": method_name,
    }


def runSingleMethod(
        item: Dict,
        method: Callable,
        lm: Any,
        prompter: Any,
        parser: Any,
        run_dir: str,
        event_sink: Any = None,
) -> float:
    """
    对单个样本执行单个方法，并保存结果。
    
    Args:
        item: 数据集中的一条样本
        method: 方法函数（返回 GraphOfOperations）
        lm: 语言模型实例
        prompter: Prompter 实例
        parser: Parser 实例
        run_dir: 运行结果目录
    
    Returns:
        cost: 本次执行的花费
    """

    problem_params = buildProblemParams(item, method.__name__)

    # # 如果是 got 方法，先调用“自动 GoO 设计器”为当前样本生成 GoO 建议并保存，
    # # 然后尝试基于建议动态构建 GoO；如果失败则回退到手写的 got()。
    # operations_graph = None
    # if method_name.startswith("got"):
    #     try:
    #         context_text = problem_params.get("context_text", "")
    #         preview = context_text[:3000] if isinstance(context_text, str) else ""
    #         sample_id = item.get("_id") or item.get("id") or problem_params.get(
    #             "question", ""
    #         )[:32]
    #         retry_feedback = ""
    #         max_attempts = 3
    #
    #         for attempt in range(1, max_attempts + 1):
    #             design = requestGotGooDesign(
    #                 lm=lm,
    #                 question=problem_params.get("question", ""),
    #                 context_preview=preview,
    #                 retry_feedback=retry_feedback,
    #             )
    #             # 保存每次尝试的设计结果，方便回溯
    #             design_path = saveDesignResult(
    #                 out_dir=os.path.join(run_dir, "auto_goo"),
    #                 method_name=f"{method_name}_try{attempt}",
    #                 sample_id=sample_id,
    #                 design=design,
    #             )
    #
    #             try:
    #                 parsed_design = loadGooDesignFromFile(design_path)
    #                 built = buildGraphFromGooDesign(
    #                     parsed_design,
    #                     scoring_function=score.scoreMultiHop,
    #                     ground_truth_evaluator=score.testMultiHop,
    #                 )
    #                 operations_graph = built.graph
    #                 logging.info(
    #                     "Using AI-designed GoO for sample %s (attempt %d)",
    #                     sample_id,
    #                     attempt,
    #                 )
    #                 break
    #             except Exception as build_err:
    #                 msg = str(build_err).lower()
    #                 if "cycle" in msg or "acyclic" in msg or "topologically" in msg:
    #                     details = str(build_err)
    #                     retry_feedback = (
    #                         "Your previous GoO contains a cycle (or cannot be topologically sorted). "
    #                         "Please regenerate a STRICT DAG with no cycles. "
    #                         "Ensure dependencies are valid and keep branch+merge structure. "
    #                         f"Cycle diagnostics from validator: {details}"
    #                     )
    #                     logging.warning(
    #                         "AI GoO attempt %d has cycle for sample %s, details: %s",
    #                         attempt,
    #                         sample_id,
    #                         details,
    #                     )
    #                     continue
    #                 # 非环错误直接抛出，外层兜底回退
    #                 raise
    #
    #         if operations_graph is None:
    #             logging.warning(
    #                 "AI GoO failed after %d attempts for sample %s, fallback to handwritten got().",
    #                 max_attempts,
    #                 sample_id,
    #             )
    #     except Exception as e:
    #         logging.warning(
    #             "Auto GoO build failed for %s, fallback to default graph: %s",
    #             method_name,
    #             e,
    #         )

    # 非 got 或 got 动态构建失败时，回退到默认手写图
    # if operations_graph is None:
    #     operations_graph = method()
    if method.__name__.startswith("multiAgentGoT"):
        operations_graph = method(problem_params.get("max_subquestions", 4))
    else:
        operations_graph = method()

    executor = controller.Controller(
        lm,
        operations_graph,
        prompter,
        parser,
        problem_params,
        event_sink=event_sink,
    )

    run_error = None
    try:
        executor.run()
    except Exception as e:
        run_error = e
        logging.error(
            "Exception in %s for item %s: %s",
            method.__name__,
            item.get("_id", ""),
            e,
            exc_info=True,
        )

    # 保存结果
    out_path = os.path.join(
        run_dir, method.__name__, f"{item.get('_id', id(item))}.json"
    )
    executor.output_graph(out_path)

    # 额外保存“精简结果”文件，便于快速查看关键指标
    summary_path = os.path.join(
        run_dir, method.__name__, f"{item.get('_id', id(item))}.summary.json"
    )
    try:
        summary = _buildCompactResultSummary(executor=executor, item=item, method_name=method.__name__)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning("Failed to write summary json for %s: %s", method.__name__, e)

    if run_error is not None:
        raise run_error

    return getattr(lm, "cost", 0.0)


def getLmConfigPath(base_dir: str = None) -> str:
    """
    获取语言模型配置文件路径。
    
    Args:
        base_dir: 基准目录，默认为当前文件所在目录
    
    Returns:
        配置文件的绝对路径
    """
    if base_dir is None:
        base_dir = os.path.dirname(__file__)

    config_path = os.path.join(
        base_dir,
        "..",
        "graph_of_thoughts",
        "language_models",
        "config.json",
    )
    return os.path.abspath(config_path)


def _collectLmUsage(lm: Any) -> Dict[str, Any]:
    """
    汇总单次推理的 token 与费用。

    单价来自各底层模型构造时加载的 config.json：
    prompt_token_cost / response_token_cost 表示每 1000 token 的美元计价；
    费用在 ChatGPT / Gemini / DeepSeek 等实现里按累计 token 实时更新（与本函数读取的 cost 一致）。
    """
    prompt_tokens = int(getattr(lm, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(lm, "completion_tokens", 0) or 0)
    cost = float(getattr(lm, "cost", 0.0) or 0.0)
    usage: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": round(cost, 8),
        "pricing_note": (
            "estimated_cost_usd 由各模型实现按 config.json 中每 1000 token 的 "
            "prompt_token_cost、response_token_cost 对累计 prompt/completion tokens 计算并求和；"
            "多角色时顶层为全部角色实例之和。"
        ),
    }
    role_map = getattr(lm, "role_to_lm", None)
    if isinstance(role_map, dict) and role_map:
        by_role: Dict[str, Any] = {}
        for role, sub_lm in role_map.items():
            pt = int(getattr(sub_lm, "prompt_tokens", 0) or 0)
            ct = int(getattr(sub_lm, "completion_tokens", 0) or 0)
            by_role[role] = {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
                "estimated_cost_usd": round(float(getattr(sub_lm, "cost", 0.0) or 0.0), 8),
            }
        usage["by_role"] = by_role
    return usage


def _buildCompactResultSummary(executor: Any, item: Dict, method_name: str) -> Dict:
    """
    从执行器中提取关键结果，生成精简版结果字典。
    包含：EM、F1、score、是否解决、AI评价、标准答案、预测答案、usage（token 与费用）等。
    """
    final_state = {}
    final_score = None
    solved = False

    # 默认取最后一个叶子节点的第一个 thought 作为最终结果
    # 对当前流程（以 GroundTruth 结尾）通常就是最终答案节点。
    if getattr(executor, "graph", None) and getattr(executor.graph, "leaves", None):
        leaves = executor.graph.leaves
        if leaves:
            thoughts = leaves[-1].get_thoughts()
            if thoughts:
                t = thoughts[0]
                final_state = t.state or {}
                solved = bool(getattr(t, "solved", False))
                try:
                    final_score = float(getattr(t, "score", 0.0))
                except Exception:
                    final_score = None

    predicted = final_state.get("answer", "")
    gold = final_state.get("ground_truth_answer", item.get("answer", ""))
    aliases = final_state.get("answer_aliases", item.get("answer_aliases", [])) or []

    em = score.answerEMScore(predicted, gold, aliases)
    f1 = score.answerF1Score(predicted, gold, aliases)

    # “AI评价”优先使用最终答案的 global_critique，其次回退到分支 critique
    ai_eval = final_state.get("global_critique", "") or final_state.get("critique", "")

    summary = {
        "id": item.get("_id", id(item)),
        "method": method_name,
        "question": item.get("question", ""),
        "ground_truth_answer": gold,
        "predicted_answer": predicted,
        "EM": bool(em),
        "F1": float(f1),
        "score": final_score,
        "problem_solved": bool(solved),
        "ai_evaluation": ai_eval,
    }
    lm = getattr(executor, "lm", None)
    if lm is not None:
        try:
            summary["usage"] = _collectLmUsage(lm)
        except Exception as e:
            logging.warning("Failed to collect LM usage for summary: %s", e)
            summary["usage"] = {
                "error": str(e),
                "pricing_note": "未能读取 token/费用，请检查语言模型是否实现 prompt_tokens、completion_tokens、cost。",
            }
    return summary


def _method_sort_key(method_name: str) -> Tuple[int, str]:
    if method_name == "io":
        return (0, method_name)
    if method_name == "cot":
        return (1, method_name)
    if method_name == "tot":
        return (2, method_name)
    if method_name == "got":
        return (3, method_name)
    if method_name.startswith("multiAgentGoT"):
        return (4, method_name)
    return (20, method_name)


def aggregate_run_summaries(run_dir: str) -> Dict[str, Any]:
    """
    扫描某次实验目录下各方法子文件夹中的 ``*.summary.json``，按方法汇总：

    - **正确率**：各题 ``problem_solved`` 比例，与 ``score.testMultiHop`` / GroundTruth 一致
      （CoT/ToT/GoT 为 EM；multiAgentGoT 为 LLM 分数 > ``solve_score_threshold``，默认 0.8）。
    - **平均 score**：各题 ``summary['score']``（最终节点上的图分数）的算术平均，仅统计非 null。
    - **总费用**：各题 ``usage.estimated_cost_usd`` 之和。

    另附 **EM_accuracy**、**mean_F1**（与金标字符串比对）供参考，与主表「正确率」定义不同。
    """
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir 不存在: {run_dir}")

    pattern = os.path.join(run_dir, "*", "*.summary.json")
    paths = glob.glob(pattern)
    by_method: Dict[str, List[str]] = {}
    for p in paths:
        method = os.path.basename(os.path.dirname(p))
        if method.startswith("."):
            continue
        by_method.setdefault(method, []).append(p)

    methods_out: Dict[str, Any] = {}
    for method in sorted(by_method.keys(), key=_method_sort_key):
        paths_m = sorted(by_method[method])
        em_hits = 0
        f1_sum = 0.0
        solved_hits = 0
        score_sum = 0.0
        n_score = 0
        n_parsed = 0
        total_cost = 0.0
        n_cost = 0
        missing_usage = 0
        parse_failed = 0
        for fp in paths_m:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    row = json.load(f)
            except Exception as e:
                logging.warning("跳过无法解析的 summary: %s (%s)", fp, e)
                parse_failed += 1
                continue
            n_parsed += 1
            if bool(row.get("EM")):
                em_hits += 1
            if bool(row.get("problem_solved")):
                solved_hits += 1
            try:
                f1v = float(row.get("F1", 0.0))
                f1_sum += f1v
            except (TypeError, ValueError):
                pass
            raw_score = row.get("score", None)
            if raw_score is not None:
                try:
                    score_sum += float(raw_score)
                    n_score += 1
                except (TypeError, ValueError):
                    pass
            usage = row.get("usage")
            if not isinstance(usage, dict) or usage.get("error"):
                missing_usage += 1
                continue
            try:
                c = float(usage.get("estimated_cost_usd", 0.0) or 0.0)
                total_cost += c
                n_cost += 1
            except (TypeError, ValueError):
                missing_usage += 1

        n = n_parsed
        methods_out[method] = {
            "num_summaries": n,
            "summary_files_on_disk": len(paths_m),
            "summary_parse_failed": parse_failed,
            "correct_rate": float(solved_hits / n) if n else 0.0,
            "problem_solved_count": solved_hits,
            "mean_leaf_score": float(score_sum / n_score) if n_score else None,
            "mean_leaf_score_n": n_score,
            "EM_count": em_hits,
            "EM_accuracy": float(em_hits / n) if n else 0.0,
            "mean_F1": float(f1_sum / n) if n else 0.0,
            "total_cost_usd": round(total_cost, 8),
            "summaries_with_usage_cost": n_cost,
            "summaries_missing_usage_or_cost": missing_usage,
        }

    return {
        "run_dir": run_dir,
        "methods": methods_out,
        "note_zh": (
            "主表「正确率」= problem_solved 为 true 的比例，与 GroundTruth 所用 testMultiHop 一致："
            "CoT/ToT/GoT 按预测与金标 EM；multiAgentGoT 按 LLM 给出的 score > solve_score_threshold（默认 0.9）。"
            "「平均score」为各题 summary 中最终节点 score 的算术平均（JSON 字段 mean_leaf_score，仅非 null 参与）。"
            "「总费用」为各题 usage.estimated_cost_usd 之和。"
            "另见 EM_accuracy、mean_F1：纯字符串 EM/F1，与 multiAgentGoT 的 LLM 阈值判定不同。"
        ),
    }


def build_aggregate_tables_json(methods: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    将汇总指标转为与终端主表、参考表一致的行列表，便于单独导出或画图。
    """
    if not methods:
        return {"primary": [], "reference_em_f1": []}
    names = sorted(methods.keys(), key=_method_sort_key)
    primary: List[Dict[str, Any]] = []
    reference: List[Dict[str, Any]] = []
    for name in names:
        m = methods[name]
        primary.append(
            {
                "method": name,
                "N": int(m.get("num_summaries", 0)),
                "correct_rate": float(m.get("correct_rate", 0.0)),
                "mean_leaf_score": m.get("mean_leaf_score"),
                "mean_leaf_score_n": int(m.get("mean_leaf_score_n", 0)),
                "total_cost_usd": float(m.get("total_cost_usd", 0.0)),
            }
        )
        reference.append(
            {
                "method": name,
                "EM_accuracy": float(m.get("EM_accuracy", 0.0)),
                "mean_F1": float(m.get("mean_F1", 0.0)),
            }
        )
    return {"primary": primary, "reference_em_f1": reference}


def print_aggregate_report_table(agg: Dict[str, Any]) -> None:
    """将 aggregate_run_summaries 的结果打印为可读表格（正确率 / 平均 score / 费用）。"""
    methods = agg.get("methods") or {}
    if not methods:
        print("\n[汇总] 未找到任何 *.summary.json，跳过表格输出。")
        return
    print("\n" + "=" * 96)
    print(" 数据集级汇总（各方法子目录下的 *.summary.json）")
    print("=" * 96)
    hdr = f"{'方法':<26} {'N':>6} {'正确率':>8} {'平均score':>12} {'总费用$':>12}"
    print(hdr)
    print("-" * 96)
    for name in sorted(methods.keys(), key=_method_sort_key):
        m = methods[name]
        n = int(m.get("num_summaries", 0))
        cr = float(m.get("correct_rate", 0.0))
        ms = m.get("mean_leaf_score")
        ms_str = f"{float(ms):.4f}" if ms is not None else "  —  "
        cost = float(m.get("total_cost_usd", 0.0))
        print(f"{name:<26} {n:>6} {cr:>8.4f} {ms_str:>12} {cost:>12.4f}")
    print("=" * 96)
    print(
        "参考（与金标字符串比对，非 multiAgentGoT 主判定）:"
    )
    ref_hdr = f"{'方法':<26} {'EM率':>8} {'平均F1':>10}"
    print(ref_hdr)
    print("-" * 48)
    for name in sorted(methods.keys(), key=_method_sort_key):
        m = methods[name]
        em_acc = float(m.get("EM_accuracy", 0.0))
        mf1 = float(m.get("mean_F1", 0.0))
        print(f"{name:<26} {em_acc:>8.4f} {mf1:>10.4f}")
    print("=" * 96)
    if agg.get("note_zh"):
        print(agg["note_zh"])
    print()


# 每完成若干道题写入一次进度汇总表（与全量总表格式一致，文件名带 progress_nXXX）
DATASET_AGGREGATE_CHECKPOINT_EVERY = 10


def finalize_run_aggregate(
    run_dir: str,
    *,
    progress_completed_n: Optional[int] = None,
    print_table: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    写入汇总 JSON 并可选打印表格。

    - 全量（``progress_completed_n`` 为 None）：``dataset_aggregate_metrics.json``、
      ``dataset_aggregate_table.json``（跑完后的总表）。
    - 进度（``progress_completed_n`` 为当前已累计完成的题目数，如 10）：
      ``dataset_aggregate_metrics_progress_n010.json``、
      ``dataset_aggregate_table_progress_n010.json``，
      统计的是此时 run_dir 下已存在的 ``*.summary.json``（部分题目）。
    """
    try:
        agg = aggregate_run_summaries(run_dir)
        agg["generated_at"] = datetime.datetime.now().replace(microsecond=0).isoformat()
        agg["tables"] = build_aggregate_tables_json(agg.get("methods") or {})

        if progress_completed_n is not None:
            agg["checkpoint_completed_samples"] = int(progress_completed_n)
            suffix = f"_progress_n{int(progress_completed_n):03d}"
            metrics_name = f"dataset_aggregate_metrics{suffix}.json"
            table_name = f"dataset_aggregate_table{suffix}.json"
        else:
            metrics_name = "dataset_aggregate_metrics.json"
            table_name = "dataset_aggregate_table.json"

        out_path = os.path.join(run_dir, metrics_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)

        table_only = {
            "run_dir": agg.get("run_dir"),
            "generated_at": agg.get("generated_at"),
            "note_zh": agg.get("note_zh"),
            "tables": agg.get("tables"),
        }
        if progress_completed_n is not None:
            table_only["checkpoint_completed_samples"] = int(progress_completed_n)
        table_path = os.path.join(run_dir, table_name)
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(table_only, f, ensure_ascii=False, indent=2)

        if print_table:
            print_aggregate_report_table(agg)
        if progress_completed_n is not None:
            logging.info(
                "进度汇总（已完成 %s 题）已写入: %s 与 %s",
                progress_completed_n,
                out_path,
                table_path,
            )
        else:
            logging.info("数据集汇总已写入: %s 与 %s", out_path, table_path)
        return agg
    except Exception as e:
        logging.warning("数据集汇总失败: %s", e, exc_info=True)
        return None
