import datetime
import json
import logging
import os
from typing import Dict, List, Callable, Any

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
        "solve_score_threshold": 0.9,
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

    try:
        executor.run()
    except Exception as e:
        logging.error(
            "Exception in %s for item %s: %s", method.__name__, item.get("_id", ""), e
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


def _buildCompactResultSummary(executor: Any, item: Dict, method_name: str) -> Dict:
    """
    从执行器中提取关键结果，生成精简版结果字典。
    包含：EM、F1、score、是否解决、AI评价、标准答案、预测答案等。
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
    return summary
