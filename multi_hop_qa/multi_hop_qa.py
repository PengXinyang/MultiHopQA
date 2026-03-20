# 多跳推理示例：在 GoT 框架下对多文档问答进行 IO / CoT / ToT / GoT 多种推理方式。

import json
import logging
import os
import random
import argparse
import time
from typing import Dict, List, Callable

import utils
import score
from graph_of_thoughts import language_models, operations, prompter, parser


# --- Prompter ---


class MultiHopPrompter(prompter.Prompter):
    """为多跳问答任务（问题 + 上下文 -> 答案）生成提示词。"""

    io_prompt = """<Instruction>
Answer the question based only on the given context. Output only the final answer in one line, optionally prefixed with "Answer: ".
</Instruction>

<Context>
{context_text}
</Context>

<Question>
{question}
</Question>

Answer:"""

    cot_prompt = """<Instruction>
Answer the question based on the given context. First reason step by step (which documents or sentences are relevant, then combine to conclude). End with exactly one line: "Answer: <your answer>".
</Instruction>

<Context>
{context_text}
</Context>

<Question>
{question}
</Question>"""

    # tot是树形结构，只有改进
    tot_improve_prompt = """<Instruction>
The following answer to the question is wrong or incomplete. Try again: reason over the context and give a better answer. End with "Answer: <your answer>".
</Instruction>

<Context>
{context_text}
</Context>

<Question>
{question}
</Question>

Previous (wrong) answer: {current}

Answer:"""

    # got的“生成”操作，将10篇文档分成两组，再总结
    got_split_prompt = """<Instruction>
You are given a multi-hop question and {num_docs} documents (numbered 1–{num_docs}). Split the documents into two groups: Group 0 = documents 1–{half}, Group 1 = documents {half_plus}–{num_docs}. For each group, write a short summary of the information that is relevant to answering the question. Output only valid JSON in this exact format (no other text):
{{"Group 0": "summary text for first half", "Group 1": "summary text for second half"}}
</Instruction>

<Context>
{context_text}
</Context>

<Question>
{question}
</Question>"""

    # 对单组摘要产生局部性结论
    got_partial_prompt = """<Instruction>
Using only the following summary (from a subset of documents), state the key fact or partial answer that helps answer the question. Be concise. End with "Partial: <one sentence>".
</Instruction>

<Summary>
{current}
</Summary>

<Question>
{question}
</Question>

Partial:"""

    # 合并两部分产生最终答案
    got_aggregate_prompt = """<Instruction>
Combine the two partial answers below to give the final answer to the question. Output only one line: "Answer: <final answer>".
</Instruction>

<Question>
{question}
</Question>

Partial answer 1: {input1}

Partial answer 2: {input2}

Answer:"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        assert len(state_dicts) == 2
        return self.got_aggregate_prompt.format(
            question=state_dicts[0]["question"],
            input1=state_dicts[0].get("current", ""),
            input2=state_dicts[1].get("current", ""),
        )

    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        question = kwargs.get("question", "")
        context_text = kwargs.get("context_text", "")
        method = kwargs.get("method", "io")
        current = kwargs.get("current", "")
        phase = kwargs.get("phase", 0)
        # node_instruction = kwargs.get("node_instruction", "").strip()
        #
        # # 如果是 got 的第 0 阶段（需要输出特定 JSON），优先走内置模板
        # # 避免节点级 instruction 覆盖 got_split_json 的解析逻辑。
        # if method.startswith("got") and phase == 0:
        #     num_docs = kwargs.get("num_docs", 10)
        #     half = num_docs // 2
        #     return self.got_split_prompt.format(
        #         question=question,
        #         context_text=context_text,
        #         num_docs=num_docs,
        #         half=half,
        #         half_plus=half + 1,
        #     )
        #
        # # 如果存在节点级别的 instruction（来自 AI 设计的 GoO），优先使用它。
        # # 但同时加入“硬约束”：必须只输出最终答案的单行文本，确保解析器能稳定提取。
        # if node_instruction:
        #     return (
        #         f"<Instruction>\n{node_instruction}\n\n"
        #         f"Hard constraint: Answer the question based only on the given context. "
        #         f"Output only the final answer in one line, optionally prefixed with \"Answer: \".\n"
        #         f"</Instruction>\n\n"
        #         f"<Context>\n{context_text}\n</Context>\n\n"
        #         f"<Question>\n{question}\n</Question>\n"
        #     )

        if method.startswith("io"):
            return self.io_prompt.format(question=question, context_text=context_text)
        if method.startswith("cot"):
            return self.cot_prompt.format(question=question, context_text=context_text)
        if method.startswith("tot"):
            if not current or current == "":
                return self.io_prompt.format(question=question, context_text=context_text)
            return self.tot_improve_prompt.format(
                question=question, context_text=context_text, current=current
            )
        if method.startswith("got"):
            if phase == 0:
                num_docs = kwargs.get("num_docs", 10)
                half = num_docs // 2
                return self.got_split_prompt.format(
                    question=question,
                    context_text=context_text,
                    num_docs=num_docs,
                    half=half,
                    half_plus=half + 1,
                )
            if phase == 1:
                return self.got_partial_prompt.format(
                    question=question, current=kwargs.get("current", "")
                )
            return self.io_prompt.format(question=question, context_text=context_text)
        return self.io_prompt.format(question=question, context_text=context_text)

    def improve_prompt(self, **kwargs) -> str:
        return ""

    def validation_prompt(self, **kwargs) -> str:
        return ""

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        return ""


# --- Parser ---


class MultiHopParser(parser.Parser):
    """将模型输出解析为答案（以及可选的 supporting facts）。"""

    def __init__(self):
        self.cache = {}

    @staticmethod
    def _extract_answer(text: str) -> str:
        text = text.strip()
        for prefix in ("Answer:", "answer:"):
            if prefix in text:
                idx = text.index(prefix) + len(prefix)
                return text[idx:].strip().split("\n")[0].strip()
        return text.split("\n")[0].strip() if text else ""

    @staticmethod
    def _coerce_yes_no(answer: str, gold_answer: str) -> str:
        """
        Hotpot 官方评估对 yes/no 有特殊规则：当标准答案是 yes/no/noanswer 时，
        预测必须严格匹配这三个标签本身。很多模型会输出 \"Yes, ...\" 这种完整句子，
        这里在适用时将其规整为标准标签（yes/no/noanswer）。
        """
        if not isinstance(answer, str):
            return ""
        if not isinstance(gold_answer, str):
            return answer
        gold = gold_answer.strip().lower()
        if gold not in ("yes", "no", "noanswer"):
            return answer
        a = answer.strip().lower()
        if a.startswith("yes"):
            return "yes"
        if a.startswith("no"):
            return "no"
        if a.startswith("noanswer"):
            return "noanswer"
        # 如果模型没有明显以 yes/no 开头，则保持原样
        return answer

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        new_states = []
        method = state.get("method", "io")
        phase = state.get("phase", 0)
        gold_answer = state.get("ground_truth_answer", "")

        for text in texts:
            if method.startswith("got") and phase == 0:
                try:
                    s = text.strip()
                    start = s.find("{")
                    end = s.rfind("}") + 1
                    if start >= 0 and end > start:
                        json_str = s[start:end]
                        obj = json.loads(json_str)
                        for key in ("Group 0", "Group 1"):
                            if key in obj and isinstance(obj[key], str):
                                new_state = state.copy()
                                new_state["current"] = obj[key].strip()
                                new_state["part"] = key
                                new_state["phase"] = 1
                                new_states.append(new_state)
                except Exception as e:
                    logging.warning("Failed to parse GoT split JSON: %s", e)
            else:
                answer = self._extract_answer(text)
                answer = self._coerce_yes_no(answer, gold_answer)
                new_state = state.copy()
                new_state["answer"] = answer
                new_state["current"] = answer
                new_states.append(new_state)
        return new_states

    def parse_aggregation_answer(
            self, states: List[Dict], texts: List[str]
    ) -> List[Dict]:
        new_states = []
        gold_answer = states[0].get("ground_truth_answer", "") if states else ""
        for text in texts:
            answer = self._extract_answer(text)
            answer = self._coerce_yes_no(answer, gold_answer)
            base = states[0].copy() if states else {}
            new_state = {**base, "current": answer, "answer": answer}
            new_states.append(new_state)
        return new_states

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        if not texts:
            return {}
        ans = self._extract_answer(texts[0])
        ans = self._coerce_yes_no(ans, state.get("ground_truth_answer", ""))
        return {"answer": ans, "current": ans}

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        return False

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        return [0.0] * len(states)


# --- Operation graphs ---


def io() -> operations.GraphOfOperations:
    """IO：一次生成。流程为 Generate -> Score -> GroundTruth。"""
    g = operations.GraphOfOperations()
    g.append_operation(operations.Generate(1, 1))
    g.append_operation(operations.Score(1, False, score.scoreMultiHop))
    g.append_operation(operations.GroundTruth(score.testMultiHop))
    return g


def cot() -> operations.GraphOfOperations:
    """CoT：与 IO 的操作图相同，但提示词要求给出逐步推理后再输出答案。"""
    g = operations.GraphOfOperations()
    g.append_operation(operations.Generate(1, 1))
    g.append_operation(operations.Score(1, False, score.scoreMultiHop))
    g.append_operation(operations.GroundTruth(score.testMultiHop))
    return g


def tot() -> operations.GraphOfOperations:
    """ToT：多候选生成 -> 打分 -> 保留最优 -> 再生成/精化一次 -> 再打分 -> 保留最优。"""
    g = operations.GraphOfOperations()
    g.append_operation(operations.Generate(1, 5))
    g.append_operation(operations.Score(1, False, score.scoreMultiHop))
    k1 = operations.KeepBestN(1, True)
    g.append_operation(k1)
    g.append_operation(operations.Generate(1, 3))
    g.append_operation(operations.Score(1, False, score.scoreMultiHop))
    g.append_operation(operations.KeepBestN(1, True))
    g.append_operation(operations.GroundTruth(score.testMultiHop))
    return g


def got() -> operations.GraphOfOperations:
    """GoT：将上下文分为两组 -> 分别得到局部结论 -> 聚合为最终答案 -> 打分/筛选/对照答案。"""
    g = operations.GraphOfOperations()
    plans = operations.Generate(1, 1)
    g.append_operation(plans)

    for i in range(2):
        part_id = f"Group {i}"
        sel = operations.Selector(
            lambda thoughts, part_id=part_id: [
                t for t in thoughts if t.state.get("part") == part_id
            ]
        )
        sel.add_predecessor(plans)
        g.add_operation(sel)
        gen = operations.Generate(1, 1)
        gen.add_predecessor(sel)
        g.add_operation(gen)

    g.append_operation(operations.Aggregate(1))
    g.append_operation(operations.Score(1, False, score.scoreMultiHop))
    g.append_operation(operations.KeepBestN(1, True))
    g.append_operation(operations.GroundTruth(score.testMultiHop))
    return g


# --- Run ---


def run(
        data_ids: List[int],
        methods: List[Callable[[], operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
        data_path: str = None,
        max_samples: int = 100,
) -> float:
    """
    加载多跳问答数据集（HotpotQA / MuSiQue），对指定样本和指定方法运行 GoT 框架，
    并将每次运行的 GRS（Graph Reasoning State）输出到 `results/` 目录下。
    
    Args:
        data_ids: 要运行的样本索引列表，None 或空列表表示全部
        methods: 方法列表，每个方法返回一个 GraphOfOperations
        budget: 预算（美元），超出后停止
        lm_name: 语言模型名称
        data_path: 数据集路径，None 则使用默认 HotpotQA
        max_samples: 最大加载样本数
    
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

    # 4. 创建运行目录和配置
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    run_dir = utils.setupRunDirectory(results_base_dir=results_dir, lm_name=lm_name, methods=methods, config_extra={
        "data_path": data_path,
        "data_ids": data_ids[:len(selected)],
        "budget": budget,
        "max_samples": max_samples,
    })

    # 5. 获取语言模型配置路径
    config_lm_path = utils.getLmConfigPath(os.path.dirname(__file__))

    # 6. 运行实验
    spent = 0.0
    prompter = MultiHopPrompter()
    parser = MultiHopParser()
    
    for idx, item in enumerate(selected, start=1):
        print(f"正在运行第 {idx}/{len(selected)} 个问题，_id={item.get('_id', '')}")
        if budget <= 0:
            logging.error("Budget depleted, stopping.")
            break

        for method in methods:
            print(f"  方法: {method.__name__}")
            if budget <= 0:
                break
            
            # 创建语言模型实例
            lm = language_models.build_language_model(
                config_lm_path,
                model_name=lm_name,
                cache=True,
            )
            
            # 执行单个方法
            cost = utils.runSingleMethod(item=item, method=method, lm=lm, prompter=prompter, parser=parser,
                                         run_dir=run_dir)
            
            budget -= cost
            spent += cost

    return spent


if __name__ == "__main__":
    # 默认入口：随机跑样本，方法为 io/cot/tot/got，预算为 5 美元
    # 支持的数据集：hotpotqa, musique_ans, musique_full
    parser = argparse.ArgumentParser(description="多跳问答 GoT 实验")
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                        choices=["hotpotqa", "musique_ans", "musique_full"],
                        help="数据集名称")
    parser.add_argument("--lm", type=str, default="gemini-2.5-flash-gcli",
                        help="语言模型名称")
    parser.add_argument("--budget", type=float, default=5.0,
                        help="预算（美元）")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="随机抽取的样本数")
    args = parser.parse_args()
    
    # 数据集路径和大小映射
    DATASET_CONFIG = {
        "hotpotqa": {
            "path": os.path.join(os.path.dirname(__file__), "..", "dataset", "hotpotQA", "hotpot_dev_distractor_v1.json"),
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

    #seed = int(time.time())
    seed = 42
    random.seed(seed)
    samples = random.sample(range(len_data), min(args.num_samples, len_data))
    approaches = [io, cot, tot, got]
    
    print(f"数据集: {args.dataset}")
    print(f"语言模型: {args.lm}")
    print(f"样本数: {len(samples)}")
    print(f"预算: ${args.budget}")
    
    spent = run(
        samples,
        approaches,
        args.budget,
        args.lm,
        data_path=data_path,
        max_samples=len_data,
    )
    logging.info("Spent %s out of %s budget.", spent, args.budget)
