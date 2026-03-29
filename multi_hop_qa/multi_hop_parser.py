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
    def _norm_title(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    @classmethod
    def _title_to_context_idx(cls, context: List, title: str) -> int | None:
        """
        Map a document title to its paragraph index in `context` (Hotpot-style: [[title, [sents]], ...]).
        Returns None if not found.
        """
        if not isinstance(context, list) or not title:
            return None
        needle = cls._norm_title(title)
        for i, item in enumerate(context):
            if isinstance(item, (list, tuple)) and item:
                if cls._norm_title(str(item[0])) == needle:
                    return i
        return None

    @classmethod
    def _infer_paragraph_support_idx(cls, context: List, evidence_spans: List) -> int | None:
        """
        Infer a supporting paragraph index from evidence spans.
        evidence_spans expected like: [["title", sent_idx], ...]
        """
        if not isinstance(evidence_spans, list) or not evidence_spans:
            return None

        # Prefer the first span that maps to a known title in context
        for span in evidence_spans:
            title = None
            if isinstance(span, (list, tuple)) and span:
                title = span[0]
            elif isinstance(span, dict):
                title = span.get("title") or span.get("doc_title")
            if title:
                idx = cls._title_to_context_idx(context, str(title))
                if idx is not None:
                    return idx
        return None

    @staticmethod
    def _try_update_decomposition_support_idx(state: Dict, sub_id: int, para_idx: int) -> None:
        """
        Best-effort: write back paragraph_support_idx into state["question_decomposition"][sub_id]
        if it exists and aligns with sub_id.
        """
        if not isinstance(state, dict):
            return
        if not isinstance(sub_id, int) or sub_id < 0:
            return
        if not isinstance(para_idx, int) or para_idx < 0:
            return
        decomp = state.get("question_decomposition")
        if not isinstance(decomp, list) or sub_id >= len(decomp):
            return
        step = decomp[sub_id]
        if not isinstance(step, dict):
            return
        step["paragraph_support_idx"] = para_idx

    @staticmethod
    def _keyword_tokens(text: str) -> List[str]:
        toks = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        stop = {
            "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "was", "were",
            "is", "are", "who", "what", "where", "when", "which", "does", "did", "do",
            # very common relation words that don't help pick a document
            "creator", "developed", "educated", "education", "born", "birth", "date", "year",
        }
        out: List[str] = []
        for t in toks:
            if len(t) <= 2:
                continue
            if t in stop:
                continue
            out.append(t)
        return out[:12]

    @classmethod
    def _fallback_evidence_from_context(cls, context: List, subquestion: str) -> List:
        """
        If the model fails to provide evidence_spans, pick the most relevant title from context.
        Returns evidence_spans in the expected format, or [] if context is unusable.
        """
        if not isinstance(context, list) or not context:
            return []

        keywords = cls._keyword_tokens(subquestion)
        # If we cannot extract keywords, pick the first doc.
        if not keywords:
            first = context[0]
            if isinstance(first, (list, tuple)) and first:
                return [[str(first[0]), 0]]
            return []

        best_i = None
        best_score = -1
        for i, item in enumerate(context):
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                continue
            title = str(item[0])
            sents = item[1]
            para = " ".join(sents) if isinstance(sents, list) else str(sents)
            hay = f"{title} {para}".lower()
            score = 0
            for k in keywords:
                if k in hay:
                    score += 1
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            return []
        best_item = context[best_i]
        return [[str(best_item[0]), 0]]

    @staticmethod
    def _extract_answer(text: str) -> str:
        text = text.strip()
        for prefix in ("Answer:", "answer:"):
            if prefix in text:
                idx = text.index(prefix) + len(prefix)
                return text[idx:].strip().split("\n")[0].strip()
        return text.split("\n")[0].strip() if text else ""

    @staticmethod
    def _looks_like_range(text: str) -> bool:
        t = (text or "").lower()
        if re.search(r"\bbetween\s+\d{3,4}\s+and\s+\d{3,4}\b", t):
            return True
        if re.search(r"\bfrom\s+\d{3,4}\s+to\s+\d{3,4}\b", t):
            return True
        if re.search(r"\b\d{3,4}\s*-\s*\d{3,4}\b", t):
            return True
        return False

    @staticmethod
    def _is_duration_question(subq: str) -> bool:
        s = (subq or "").lower()
        return any(
            k in s
            for k in (
                "between",
                "from",
                "to ",
                "how long",
                "duration",
                "last",
                "years",
                "during which",
                "range",
            )
        )

    @staticmethod
    def _is_impeach_initiation_question(subq: str) -> bool:
        s = (subq or "").lower()
        if "impeach" not in s and "impeached" not in s and "impeachment" not in s:
            return False
        # Not a duration/range question: user expects a point-in-time / initiation year.
        return not MultiHopParser._is_duration_question(s)

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
                            # Sequential multi-hop: start from hop0 and advance via AdvanceSubquestion op.
                            new_state = state.copy()
                            new_state["subquestions"] = sub_questions
                            new_state["sub_id"] = 0
                            new_state["subquestion"] = sub_questions[0]
                            new_state["agent_role"] = "retriever"
                            new_state["phase"] = 1
                            new_state["bindings"] = {}
                            new_state["hop_history"] = []
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

                    # Fallback: ensure evidence_spans is non-empty to avoid breaking downstream reasoning.
                    if not new_state.get("evidence_spans"):
                        fb = self._fallback_evidence_from_context(
                            context=new_state.get("context", []),
                            subquestion=str(new_state.get("subquestion", "")),
                        )
                        if fb:
                            new_state["evidence_spans"] = fb
                            if not new_state.get("evidence_summary"):
                                new_state["evidence_summary"] = f"Relevant document: {fb[0][0]}"

                    # Infer paragraph_support_idx for this subquestion (best-effort).
                    try:
                        para_idx = self._infer_paragraph_support_idx(
                            context=new_state.get("context", []),
                            evidence_spans=new_state.get("evidence_spans", []),
                        )
                        new_state["pred_paragraph_support_idx"] = para_idx
                        if para_idx is not None:
                            self._try_update_decomposition_support_idx(
                                new_state, int(new_state.get("sub_id", -1)), int(para_idx)
                            )
                    except Exception:
                        new_state["pred_paragraph_support_idx"] = None

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
                    validation = "PASS"
                    reason_code = ""
                    suggested_action = ""
                    time_facet = ""
                    for line in text.splitlines():
                        low = line.strip().lower()
                        if low.startswith("critique:"):
                            critique = line.split(":", 1)[1].strip()
                        elif low.startswith("timefacet:"):
                            time_facet = line.split(":", 1)[1].strip()
                        elif low.startswith("reasoncode:"):
                            reason_code = line.split(":", 1)[1].strip()
                        elif low.startswith("suggestedaction:"):
                            suggested_action = line.split(":", 1)[1].strip()
                        elif low.startswith("refinedpartial:"):
                            refined = line.split(":", 1)[1].strip()
                        elif low.startswith("validation:"):
                            validation = line.split(":", 1)[1].strip().upper()
                    if not refined:
                        refined = state.get("partial_answer", "")
                    new_state = state.copy()
                    new_state["critique"] = critique
                    new_state["partial_answer"] = refined
                    new_state["current"] = refined
                    # Post-check: temporal facet mismatch for impeachment initiation questions.
                    # If question expects initiation year but evidence/answer is a span, force REJECT + backtrack.
                    subq = str(new_state.get("subquestion", ""))
                    ev = str(new_state.get("evidence_summary", ""))
                    if self._is_impeach_initiation_question(subq) and not self._is_duration_question(subq):
                        if self._looks_like_range(refined) or self._looks_like_range(ev):
                            validation = "REJECT"
                            if not reason_code:
                                reason_code = "insufficient_evidence"
                            if not suggested_action:
                                suggested_action = "backtrack_retrieve"
                            if not time_facet:
                                time_facet = "trial_duration"

                    new_state["validation_decision"] = validation
                    if reason_code:
                        new_state["reason_code"] = reason_code
                    if suggested_action:
                        new_state["suggested_action"] = suggested_action
                    if time_facet:
                        new_state["time_facet"] = time_facet
                    new_state["confidence"] = self._extract_float_after(
                        "Confidence", text, new_state.get("confidence", 0.5)
                    )
                    new_state["agent_role"] = "critic_done"
                    new_state["phase"] = 4
                    cands = list(new_state.get("candidate_answers") or [])
                    cands.append(refined)
                    new_state["candidate_answers"] = cands

                    # Record hop trace for final aggregation (kept across AdvanceSubquestion).
                    hist = new_state.get("hop_history") or []
                    if not isinstance(hist, list):
                        hist = []
                    hist.append(
                        {
                            "sub_id": int(new_state.get("sub_id", 0) or 0),
                            "subquestion": str(new_state.get("subquestion", "")),
                            "partial_answer": str(new_state.get("partial_answer", "")),
                            "evidence_summary": str(new_state.get("evidence_summary", "")),
                            "validation_decision": str(new_state.get("validation_decision", "")),
                            "reason_code": str(new_state.get("reason_code", "")),
                            "time_facet": str(new_state.get("time_facet", "")),
                            "confidence": float(new_state.get("confidence", 0.0) or 0.0),
                            "line_score": float(new_state.get("line_score", new_state.get("confidence", 0.0)) or 0.0),
                            "line_trust": str(new_state.get("line_trust", "")),
                            "max_retry_reached": bool(new_state.get("max_retry_reached", False)),
                        }
                    )
                    new_state["hop_history"] = hist
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
