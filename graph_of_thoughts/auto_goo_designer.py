"""
自动 GoO 设计器：利用大模型，根据一个多跳问题和现有示例，自动生成 GoT 的 GoO 设计建议。

设计目标：
- 作为一个“元推理”模块，不直接参与任务推理，而是为人类或上层脚本提供 GoO 结构建议；
- 复用现有的语言模型抽象接口（AbstractLanguageModel），避免重复封装；
- 将提示词、示例和模型回复统一保存，便于后续分析与人工微调 GoO 结构。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from graph_of_thoughts.language_models import AbstractLanguageModel


@dataclass
class GoOExample:
    """
    用于在提示词中展示的 GoO 示例。

    attributes:
        name: 示例名称（如 "multi_hop_qa.got"）
        description: 示例用途的自然语言说明
        code_snippet: 构造 GraphOfOperations 的关键 Python 代码片段
    """

    name: str
    description: str
    code_snippet: str


def loadFileText(path: str, max_chars: int = 4000) -> str:
    """
    读取文件的一部分内容（避免提示词过长）。

    :param path: 文件路径
    :param max_chars: 最大读取字符数
    :return: 文本内容（可能被截断）
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read(max_chars)
        if len(text) == max_chars:
            text += "\n# ... (truncated) ..."
        return text
    except Exception as e:
        return f"# Failed to load file {path}: {e}"


def buildGooExamples(base_dir: Optional[str] = None) -> List[GoOExample]:
    """
    从当前仓库中构造若干 GoO 示例，用于作为提示词的一部分。

    当前内置示例：
        - examples 文件夹下的多个官方示例（如 sorting / doc_merge）

    :param base_dir: 仓库根目录（可选），默认为 auto_goo_designer 所在目录的上两级
    :return: GoOExample 列表
    """
    if base_dir is None:
        # graph_of_thoughts/auto_goo_designer.py -> 项目根目录
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    examples: List[GoOExample] = []

    # 使用 examples 目录中的样例，而不是 multi_hop_qa。
    # 这里优先选取与 GoO 构造相关、且你项目中常用的示例文件。
    candidate_examples = [
        (
            "examples.sorting.sorting_032",
            "排序任务中的 GraphOfOperations 构造与多阶段操作组织方式。",
            os.path.join(base_dir, "examples", "sorting", "sorting_032.py"),
        ),
        (
            "examples.doc_merge.doc_merge",
            "文档合并任务中的 GraphOfOperations 构造与聚合策略。",
            os.path.join(base_dir, "examples", "doc_merge", "doc_merge.py"),
        ),
        (
            "examples.keyword_counting.keyword_counting",
            "关键词计数任务中的操作图构造（用于补充不同任务风格）。",
            os.path.join(base_dir, "examples", "keyword_counting", "keyword_counting.py"),
        ),
        (
            "examples.set_intersection.set_intersection",
            "集合求交任务中的操作图构造与筛选流程。",
            os.path.join(base_dir, "examples", "set_intersection", "set_intersection.py"),
        ),
    ]

    for name, description, file_path in candidate_examples:
        code = loadFileText(file_path)
        examples.append(
            GoOExample(
                name=name,
                description=description,
                code_snippet=code,
            )
        )

    return examples


def buildDesignerPrompt(
    question: str,
    context_preview: str,
    examples: List[GoOExample],
    retry_feedback: str = "",
) -> str:
    """
    构造给大模型的提示词：输入一个多跳问题 + 上下文预览 + 若干 GoO 示例，
    让大模型设计一个新的 GoT 操作图方案。

    :param question: 数据集中的一个问题文本
    :param context_preview: 上下文预览（可以是若干段落拼接）
    :param examples: GoO 示例列表
    :return: 完整提示词字符串
    """
    examples_text_parts: List[str] = []
    for ex in examples:
        examples_text_parts.append(
            f"### Example: {ex.name}\n"
            f"Description:\n{ex.description}\n\n"
            f"Code snippet (Python):\n```python\n{ex.code_snippet}\n```"
        )
    examples_block = "\n\n".join(examples_text_parts)

    prompt = f"""
<Instruction>
You are an expert in Graph of Thoughts (GoT) for multi-step reasoning.

Given:
1. One multi-hop QA question and a preview of its context.
2. Several existing examples of how GraphOfOperations (GoO) are constructed in Python.

Your task:
- Propose a NEW GoT operation graph (GoO) tailored to this QA task.
- Use the operations available in this framework:
  - Generate, Score, KeepBestN, Aggregate, ValidateAndImprove, Improve, KeepValid,
    GroundTruth, Selector, GraphOfOperations.
- The output MUST be a pure JSON object with the following schema:
  {{
    "comment": "high level natural language explanation (1-3 sentences)",
    "nodes": [
      {{"id": "plans", "type": "Generate"}},
      {{"id": "score1", "type": "Score"}},
      ...
    ],
    "edges": [
      {{"from": "plans", "to": "score1"}},
      ...
    ],
    "final_nodes": ["id_of_final_node_1", "id_of_final_node_2"]
  }}

Requirements:
- The graph must be acyclic.
- The graph MUST be a DAG (Directed Acyclic Graph): absolutely no cycle is allowed.
- Make sure every node (except roots) has at least one predecessor.
- You MUST include a terminal GroundTruth node in "nodes" (type must be exactly "GroundTruth").
- The GroundTruth node should evaluate the final answer path and should be reachable from the final answer generation node.
- "final_nodes" must include the GroundTruth node id.
- IMPORTANT: Prefer a branched-and-merged GoO, not a single linear chain.
- The graph should contain at least one branching structure (a node with >=2 successors or >=2 roots).
- The graph should contain at least one merge point (a node with >=2 predecessors), preferably using Aggregate.
- If you propose only a single chain, it will be considered low quality.
</Instruction>

<Question>
{question}
</Question>

<ContextPreview>
{context_preview}
</ContextPreview>

<ExistingGoOExamples>
{examples_block}
</ExistingGoOExamples>

<RetryFeedback>
{retry_feedback}
</RetryFeedback>

<OutputFormat>
Return ONLY one JSON object, no extra explanation, no markdown fences.
</OutputFormat>
"""
    return prompt.strip()


def requestGotGooDesign(
    lm: AbstractLanguageModel,
    question: str,
    context_preview: str,
    examples: Optional[List[GoOExample]] = None,
    num_samples: int = 1,
    retry_feedback: str = "",
) -> Dict[str, Any]:
    """
    调用语言模型，请其为给定问题设计 GoT 的 GoO。

    返回值是一个字典：
        {
            "raw_text": <模型原始回复>,
            "parsed": <尽可能解析成的 JSON 对象，失败则为 {}>,
        }
    """
    if examples is None:
        examples = buildGooExamples()

    prompt = buildDesignerPrompt(
        question,
        context_preview,
        examples,
        retry_feedback=retry_feedback,
    )
    responses = lm.get_response_texts(lm.query(prompt, num_responses=num_samples))
    raw = responses[0] if responses else ""

    parsed: Dict[str, Any] = {}
    try:
        # 尝试从文本中提取 JSON 对象
        s = raw.strip()
        start = s.find("{")
        end = s.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = s[start:end]
            parsed = json.loads(json_str)
    except Exception:
        parsed = {}

    return {"raw_text": raw, "parsed": parsed, "prompt": prompt}


def saveDesignResult(
    out_dir: str,
    method_name: str,
    sample_id: Any,
    design: Dict[str, Any],
) -> str:
    """
    将 GoO 设计结果保存到指定目录下，文件名中包含方法名与样本 id。

    :param out_dir: 输出目录（通常是 multi_hop_qa 的 run_dir 子目录）
    :param method_name: 方法名称（如 "got"）
    :param sample_id: 样本标识（如数据集中的 _id 或索引）
    :param design: request_got_goo_design 返回的字典
    :return: 写入的文件路径
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = f"auto_goo_{method_name}_{sample_id}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(design, f, indent=2, ensure_ascii=False)
    return fpath

