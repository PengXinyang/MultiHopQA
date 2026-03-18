import datetime
import json
import logging
import os
from typing import Dict, List, Callable, Any

from data_loader import loadDataset, DatasetType
from graph_of_thoughts import controller


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
        results_base_dir: str,
        lm_name: str,
        methods: List[Callable],
        config_extra: Dict = None,
) -> str:
    """
    创建本次运行的结果目录，保存配置文件，设置日志。
    
    Args:
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
    run_dir = os.path.join(results_base_dir, f"{lm_name}_{method_names}_{timestamp}")
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

    return {
        "question": item.get("question", ""),
        "context": context,
        "context_text": context_text,
        "num_docs": num_docs,
        "answer": "",
        "current": "",
        "ground_truth_answer": item.get("answer", ""),
        "ground_truth_sp": item.get("supporting_facts", []),
        "answer_aliases": item.get("answer_aliases", []),
        "answerable": item.get("answerable", True),
        "question_decomposition": item.get("question_decomposition", []),
        "phase": 0,
        "method": method_name,
    }


def runSingleMethod(
        item: Dict,
        method: Callable,
        lm: Any,
        prompter: Any,
        parser: Any,
        run_dir: str,
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
    operations_graph = method()

    executor = controller.Controller(
        lm,
        operations_graph,
        prompter,
        parser,
        problem_params,
    )

    try:
        executor.run()
    except Exception as e:
        logging.error("Exception in %s for item %s: %s", method.__name__, item.get("_id", ""), e)

    # 保存结果
    out_path = os.path.join(
        run_dir, method.__name__, f"{item.get('_id', id(item))}.json"
    )
    executor.output_graph(out_path)

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
