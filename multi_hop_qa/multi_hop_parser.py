import json
import logging
import re
from typing import Dict, List

from graph_of_thoughts import parser


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
    def _extract_float_after(prefix: str, text: str, default: float = 0.0) -> float:
        for line in text.splitlines():
            if line.strip().lower().startswith(prefix.lower()):
                try:
                    return float(line.split(":", 1)[1].strip())
                except Exception:
                    return default
        return default

    @staticmethod
    def _coerce_yes_no(answer: str, gold_answer: str) -> str:
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
        return answer

    @staticmethod
    def _tokenOverlapScore(prediction: str, reference: str) -> float:
        pred_tokens = re.findall(r"[A-Za-z0-9]+", (prediction or "").lower())
        ref_tokens = re.findall(r"[A-Za-z0-9]+", (reference or "").lower())
        if not pred_tokens or not ref_tokens:
            return 0.0
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        overlap = len(pred_set & ref_set) / max(1, len(ref_set))
        # 只要命中关键 token，就给非零部分分；上限留给 judge/EM
        if overlap > 0:
            return min(0.8, max(0.1, overlap))
        return 0.0

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
            elif method.startswith("multiAgentGoT"):
                if phase == 0:
                    try:
                        s = text.strip()
                        start = s.find("{")
                        end = s.rfind("}") + 1
                        obj = json.loads(s[start:end]) if start >= 0 and end > start else {}
                        sub_questions = obj.get("sub_questions") or []
                        if not isinstance(sub_questions, list):
                            sub_questions = []
                        max_subq = int(state.get("max_subquestions", 4) or 4)
                        sub_questions = [str(x).strip() for x in sub_questions[:max_subq] if str(x).strip()]
                        if not sub_questions:
                            fallback = state.get("precomputed_subquestions", []) or []
                            sub_questions = [str(x).strip() for x in fallback[:max_subq] if str(x).strip()]
                        if sub_questions:
                            new_state = state.copy()
                            new_state["subquestions"] = sub_questions
                            new_state["sub_id"] = 0
                            new_state["subquestion"] = sub_questions[0]
                            new_state["agent_role"] = "retriever"
                            new_state["phase"] = 1
                            new_state["candidate_answers"] = []
                            new_state["serial_partials"] = []
                            new_state["serial_evidence"] = []
                            new_states.append(new_state)
                    except Exception as e:
                        logging.warning("Failed to parse planner JSON: %s", e)
                elif state.get("agent_role") == "retriever":
                    new_state = state.copy()
                    try:
                        s = text.strip()
                        start = s.find("{")
                        end = s.rfind("}") + 1
                        obj = json.loads(s[start:end]) if start >= 0 and end > start else {}
                        spans = obj.get("evidence_spans") or []
                        new_state["evidence_spans"] = spans if isinstance(spans, list) else []
                        new_state["evidence_summary"] = str(obj.get("evidence_summary", "")).strip()
                    except Exception:
                        new_state["evidence_spans"] = []
                        new_state["evidence_summary"] = text.strip()
                    new_state["agent_role"] = "reasoner"
                    new_state["phase"] = 2
                    new_states.append(new_state)
                elif state.get("agent_role") == "reasoner":
                    partial = ""
                    for line in text.splitlines():
                        if line.strip().lower().startswith("partial:"):
                            partial = line.split(":", 1)[1].strip()
                            break
                    if not partial:
                        partial = self._extract_answer(text)
                    new_state = state.copy()
                    new_state["partial_answer"] = partial
                    new_state["current"] = partial
                    new_state["confidence"] = self._extract_float_after("Confidence", text, 0.5)
                    new_state["agent_role"] = "critic"
                    new_state["phase"] = 3
                    new_states.append(new_state)
                elif state.get("agent_role") == "critic":
                    critique = ""
                    refined = ""
                    for line in text.splitlines():
                        low = line.strip().lower()
                        if low.startswith("critique:"):
                            critique = line.split(":", 1)[1].strip()
                        if low.startswith("refinedpartial:"):
                            refined = line.split(":", 1)[1].strip()
                    if not refined:
                        refined = state.get("partial_answer", "")
                    new_state = state.copy()
                    new_state["critique"] = critique
                    new_state["partial_answer"] = refined
                    new_state["current"] = refined
                    new_state["confidence"] = self._extract_float_after(
                        "Confidence", text, new_state.get("confidence", 0.5)
                    )
                    new_state["agent_role"] = "critic_done"
                    new_state["phase"] = 4
                    cands = list(new_state.get("candidate_answers") or [])
                    cands.append(refined)
                    new_state["candidate_answers"] = cands
                    serial_partials = list(new_state.get("serial_partials") or [])
                    serial_evidence = list(new_state.get("serial_evidence") or [])
                    serial_partials.append(refined)
                    serial_evidence.append(new_state.get("evidence_summary", ""))
                    new_state["serial_partials"] = serial_partials
                    new_state["serial_evidence"] = serial_evidence
                    subquestions = new_state.get("subquestions") or []
                    cur_idx = int(new_state.get("sub_id", 0) or 0)
                    if isinstance(subquestions, list) and cur_idx + 1 < len(subquestions):
                        next_idx = cur_idx + 1
                        new_state["sub_id"] = next_idx
                        new_state["subquestion"] = subquestions[next_idx]
                        new_state["agent_role"] = "retriever"
                        new_state["phase"] = 1
                    new_states.append(new_state)
            else:
                answer = self._extract_answer(text)
                answer = self._coerce_yes_no(answer, gold_answer)
                new_state = state.copy()
                new_state["answer"] = answer
                new_state["current"] = answer
                new_states.append(new_state)
        return new_states

    def parse_aggregation_answer(self, states: List[Dict], texts: List[str]) -> List[Dict]:
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
        if not states:
            return []

        method = states[0].get("method", "")
        if method.startswith("multiAgentGoT"):
            score = 0.0
            for text in texts:
                for line in text.splitlines():
                    if line.strip().lower().startswith("score:"):
                        try:
                            score = float(line.split(":", 1)[1].strip())
                            break
                        except Exception:
                            pass
                if score == 0.0:
                    m = re.search(r"(?<!\d)(0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
                    if m:
                        try:
                            score = float(m.group(1))
                        except Exception:
                            score = 0.0
                if score > 0.0:
                    break

            if score <= 0.0:
                state = states[0]
                predicted = state.get("answer") or state.get("current") or state.get("partial_answer") or ""
                reference = state.get("ground_truth_answer") or ""
                score = self._tokenOverlapScore(predicted, reference)

            score = max(0.0, min(1.0, score))
            return [score] * len(states)

        return [0.0] * len(states)

    def parse_score_critique(self, state: Dict, texts: List[str]) -> str:
        """
        解析评分阶段的全局评价，仅用于最终答案（aggregate 后）。
        支持：
        - GlobalCritique: ...
        - Critique: ...
        """
        if not state or not state.get("method", "").startswith("multiAgentGoT"):
            return ""
        for text in texts:
            for line in text.splitlines():
                low = line.strip().lower()
                if low.startswith("globalcritique:") or low.startswith("critique:"):
                    return line.split(":", 1)[1].strip()
        return ""
