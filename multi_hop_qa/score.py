import re
import string
from collections import Counter
from typing import Dict, List


# --- 答案归一化与 F1 计算（对齐 Hotpot 官方评测） ---


def normalizeAnswer(s: str) -> str:
    def removeArticles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def whiteSpaceFix(text):
        return " ".join(text.split())

    def removePunc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return lower(removePunc(whiteSpaceFix(removeArticles(s))))


def singleF1(prediction: str, ground_truth: str) -> float:
    """计算单个预测与标准答案的 F1 分数。"""
    np = normalizeAnswer(prediction)
    ng = normalizeAnswer(ground_truth)
    # HotpotQA 的 yes/no 特殊处理
    if np in ("yes", "no", "noanswer") and np != ng:
        return 0.0
    if ng in ("yes", "no", "noanswer") and np != ng:
        return 0.0
    pred_tok = np.split()
    gold_tok = ng.split()
    common = Counter(pred_tok) & Counter(gold_tok)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tok)
    recall = num_same / len(gold_tok)
    return 2 * precision * recall / (precision + recall)


def answerF1Score(prediction: str, ground_truth: str, answer_aliases: List[str] = None) -> float:
    """
    计算答案的 F1 分数（数值越大越好）。

    如果提供了 answer_aliases（MuSiQue 数据集），会计算与所有可接受答案的 F1，取最大值。
    """
    # 计算与主答案的 F1
    best_f1 = singleF1(prediction, ground_truth)

    # 如果有答案别名，也计算并取最大值
    if answer_aliases:
        for alias in answer_aliases:
            f1 = singleF1(prediction, alias)
            if f1 > best_f1:
                best_f1 = f1

    return best_f1


def answerEMScore(prediction: str, ground_truth: str, answer_aliases: List[str] = None) -> bool:
    """
    判断预测答案是否与标准答案完全匹配（Exact Match）。

    如果提供了 answer_aliases（MuSiQue 数据集），只要匹配任一即可。
    """
    np = normalizeAnswer(prediction)
    if np == normalizeAnswer(ground_truth):
        return True
    if answer_aliases:
        for alias in answer_aliases:
            if np == normalizeAnswer(alias):
                return True
    return False


# supporting_facts：形如 [title, sent_idx] 的列表；这里按 (title, idx) 组成的集合比较
def supportingFactsF1Score(prediction: List[List], gold: List[List]) -> float:
    pred_set = set(map(tuple, prediction))
    gold_set = set(map(tuple, gold))
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def combinedScore(state: Dict) -> float:
    """用于排序的单一分数：0.7 * 答案 F1 + 0.3 * 证据 F1（若提供 supporting_facts）。"""
    ans = state.get("answer") or ""
    gold_ans = state.get("ground_truth_answer") or ""
    answer_aliases = state.get("answer_aliases") or []
    a_f1 = answerF1Score(ans, gold_ans, answer_aliases)
    gold_sp = state.get("ground_truth_sp") or []
    pred_sp = state.get("supporting_facts")
    if isinstance(pred_sp, list) and len(pred_sp) > 0 and len(gold_sp) > 0:
        s_f1 = supportingFactsF1Score(pred_sp, gold_sp)
        return 0.7 * a_f1 + 0.3 * s_f1
    return a_f1

# --- GroundTruth 判定与给 GraphOfOperations 使用的打分函数 ---


def testMultiHop(state: Dict) -> bool:
    """
    GroundTruth 判定逻辑：
    - 其它方法：没有大模型评分，所以使用 EM（严格匹配）
    - multiAgentGoT：使用评分阈值（score > threshold 视为 solved）
    """
    try:
        method = (state.get("method") or "").strip()
        if method.startswith("multiAgentGoT"):
            threshold = state.get("solve_score_threshold", 0.9)
            try:
                threshold = float(threshold)
            except Exception:
                threshold = 0.9
            raw_score = state.get("_thought_score", state.get("score", 0.0))
            try:
                judge_score = float(raw_score)
            except Exception:
                judge_score = 0.0
            return judge_score > threshold

        ans = state.get("answer") or ""
        gold = state.get("ground_truth_answer") or ""
        answer_aliases = state.get("answer_aliases") or []
        return answerEMScore(ans, gold, answer_aliases)
    except Exception:
        return False


def numErrorsMultiHop(state: Dict) -> float:
    """给 Score 操作用的“误差分”：返回 1 - 答案 F1（0 表示完全正确，1 表示最差）。"""
    try:
        ans = state.get("answer") or ""
        gold = state.get("ground_truth_answer") or ""
        answer_aliases = state.get("answer_aliases") or []
        f1 = answerF1Score(ans, gold, answer_aliases)
        return 1.0 - f1  # num_errors: 0 = perfect, 1 = worst
    except Exception:
        return 1.0


def scoreMultiHop(state: Dict) -> float:
    """给 Score 操作用的打分函数：值越大越好（答案 F1，可叠加 supporting_facts F1）。"""
    return combinedScore(state)


def _computeOnlineReasoningScore(state: Dict) -> float:
    """
    在线推理评分（不依赖 gold）：
    - 证据覆盖：evidence_spans 数量相对 hop 数的覆盖比例
    - 角色置信：confidence 字段
    - 一致性：候选答案投票一致度（若提供）
    """
    evidence_spans = state.get("evidence_spans") or []
    num_hops = state.get("num_hops") or 2
    try:
        evidence_score = min(1.0, len(evidence_spans) / max(1, int(num_hops)))
    except Exception:
        evidence_score = 0.0

    confidence = state.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence = 0.0

    candidates = state.get("candidate_answers") or []
    consistency = 0.0
    if isinstance(candidates, list) and candidates:
        normed = [normalizeAnswer(str(x)) for x in candidates if str(x).strip()]
        if normed:
            counts = Counter(normed)
            consistency = max(counts.values()) / len(normed)

    return 0.45 * evidence_score + 0.35 * confidence + 0.20 * consistency


def scoreMultiAgentGoT(state: Dict) -> float:
    """
    multiAgentGoT 混合评分：在线推理 + 离线指标。

    在线推理用于搜索阶段可用信号，离线指标用于实验评估对齐。
    组合权重可按实验需求微调。
    """
    online = _computeOnlineReasoningScore(state)
    offline = combinedScore(state)
    state["online_reasoning_score"] = online
    state["offline_metric_score"] = offline
    return 0.6 * online + 0.4 * offline
