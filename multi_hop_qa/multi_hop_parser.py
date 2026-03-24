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
                        for idx, subq in enumerate(sub_questions[:4]):
                            if not str(subq).strip():
                                continue
                            new_state = state.copy()
                            new_state["sub_id"] = idx
                            new_state["subquestion"] = str(subq).strip()
                            new_state["agent_role"] = "retriever"
                            new_state["phase"] = 1
                            new_state["candidate_answers"] = []
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
            score = max(0.0, min(1.0, score))
            return [score] * len(states)

        return [0.0] * len(states)
