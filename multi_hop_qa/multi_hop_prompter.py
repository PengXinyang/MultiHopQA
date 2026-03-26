from typing import Dict, List

from graph_of_thoughts import prompter


class MultiHopPrompter(prompter.Prompter):
    """为多跳问答任务（问题 + 上下文 -> 答案）生成提示词。"""

    io_prompt = """<Instruction>
Answer the question based only on the given context. Output only the final answer in one line, optionally prefixed with "Answer: ".
If the answer is definite and can be given as a single token or minimal phrase (e.g. a number like "fifteen" or "15", a name, yes/no, a nationality, a date), output ONLY that—do not wrap it in a full sentence.
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
If the answer is definite and can be given as a single token or minimal phrase (number, name, yes/no, nationality, date, etc.), put ONLY that after "Answer:"—not a full sentence.
</Instruction>

<Context>
{context_text}
</Context>

<Question>
{question}
</Question>"""

    tot_improve_prompt = """<Instruction>
The following answer to the question is wrong or incomplete. Try again: reason over the context and give a better answer. End with "Answer: <your answer>".
If the correct answer is definite and can be a single token or minimal phrase (number, name, yes/no, etc.), output only that after "Answer:".
</Instruction>

<Context>
{context_text}
</Context>

<Question>
{question}
</Question>

Previous (wrong) answer: {current}

Answer:"""

    got_split_prompt = """<Instruction>
You are given a multi-hop question and {num_docs} documents (numbered 1–{num_docs}). Two overlapping ranges (boundary docs may appear in BOTH summaries):
- Group 0: documents {g0_start}–{g0_end}
- Group 1: documents {g1_start}–{g1_end}

For each group, write a short summary of the information in THAT range that is relevant to answering the question. Output only valid JSON in this exact format (no other text):
{{"Group 0": "summary text for Group 0 range", "Group 1": "summary text for Group 1 range"}}
</Instruction>

<Context>
{context_text}
</Context>

<Question>
{question}
</Question>"""

    got_partial_prompt = """<Instruction>
Using only the following summary (from a subset of documents), state the key fact or partial answer that helps answer the question. Be concise. End with "Partial: ...".
If the fact is a definite single value (number, name, yes/no, etc.), write only that value after "Partial:" (no need for a full sentence).
Otherwise one short sentence is OK.
</Instruction>

<Summary>
{current}
</Summary>

<Question>
{question}
</Question>

Partial:"""

    got_aggregate_prompt = """<Instruction>
Combine the two partial answers below to give the final answer to the question. Output only one line: "Answer: <final answer>".
If the combined answer is definite and can be expressed as a single token or minimal phrase (number, name, yes/no, nationality, etc.), output ONLY that after "Answer:"—do not use a full sentence.
</Instruction>

<Question>
{question}
</Question>

Partial answer 1: {input1}

Partial answer 2: {input2}

Answer:"""

    ma_planner_prompt = """<Instruction>
You are the Planner agent in a 4-agent multi-hop QA system.
Decompose the question into at most {max_subquestions} executable sub-questions.
Output ONLY valid JSON:
{{"sub_questions": ["...", "...", "..."]}}
</Instruction>

<Question>
{question}
</Question>

<Context>
{context_text}
</Context>"""

    ma_retriever_prompt = """<Instruction>
You are the Retriever agent. For the given SubQuestion, you MUST extract evidence from the provided Context.

Hard constraints:
- Output ONLY valid JSON (no markdown, no extra text).
- The JSON MUST contain a NON-EMPTY "evidence_spans" list.
- Each evidence span MUST be in the form ["title", sent_idx].
- "title" MUST match EXACTLY one of the document titles in the Context (the text inside [...] headers).
- If you cannot identify the exact sentence index, use sent_idx = 0 (do NOT output null).
- Prefer spans that DIRECTLY support answering the SubQuestion.
- If the answer is not explicitly stated, still choose the single MOST RELEVANT document title as evidence (do not leave evidence_spans empty).
- If <Bindings> is non-empty, you MUST prioritize evidence that mentions (or directly connects to) the bound entity/value(s).

Return ONLY JSON in this exact format:
{{"evidence_spans":[["title", sent_idx], ...], "evidence_summary":"one short evidence summary grounded in the cited span(s)"}}
</Instruction>

<Bindings>
{bindings_text}
</Bindings>

<SubQuestion>
{subquestion}
</SubQuestion>

<Context>
{context_text}
</Context>"""

    ma_reasoner_prompt = """<Instruction>
You are the Reasoner agent. Use only provided evidence summary to produce a partial answer.
If <Bindings> is non-empty, your partial answer MUST be about the bound entity/value(s). If the evidence does not support that, output:
Partial: NEED_RETRIEVE
Confidence: 0.0
Return exactly:
Partial: <partial answer>
Confidence: <0.0-1.0>
</Instruction>

<Question>
{question}
</Question>

<Bindings>
{bindings_text}
</Bindings>

<SubQuestion>
{subquestion}
</SubQuestion>

<Evidence>
{evidence_summary}
</Evidence>"""

    ma_critic_prompt = """<Instruction>
You are the Critic agent. Judge and refine the partial answer.
You MUST also VERIFY if the extracted evidence and partial answer are relevant and correct for the given subquestion.
If they are irrelevant, hallucinated, or logically wrong, you MUST output "Validation: REJECT".
If they are helpful and correct, output "Validation: PASS".
Return exactly:
Critique: <short judgement>
ReasonCode: <one of: wrong_entity, insufficient_evidence, not_grounded, other>
SuggestedAction: <one of: backtrack_retrieve, backtrack_reason, accept>
Validation: <PASS or REJECT>
RefinedPartial: <better partial answer>
Confidence: <0.0-1.0>
</Instruction>

<Question>
{question}
</Question>

<SubQuestion>
{subquestion}
</SubQuestion>

<PartialAnswer>
{partial_answer}
</PartialAnswer>

<Evidence>
{evidence_summary}
</Evidence>"""

    ma_aggregate_prompt = """<Instruction>
You are the final aggregator in multi-agent GoT.
Fuse all refined partial answers and output exactly one line:
Answer: <final answer>
Hard constraints:
- Do NOT introduce any new entity that does not appear in the provided partial answers or evidence summaries.
- If multiple entities appear, choose only the entity that best answers the question.
- Keep the answer minimal (entity/phrase), not a long explanation.
- Temporal precision rule (IMPORTANT): if the question is asking "when"/a date/time AND the provided partials/evidence contain a more specific date than just a year, you MUST keep the finest supported granularity.
  Examples:
  - If you see "April 2012" anywhere relevant, do NOT answer only "2012"; answer "April 2012".
  - If you see "7 January 2011", do NOT answer "2011"; answer "7 January 2011".
  - Only output a bare year (e.g., "2012") if month/day are NOT present in the provided partials/evidence.
</Instruction>

<Question>
{question}
</Question>

<Partials>
{partials_text}
</Partials>"""

    ma_score_prompt = """<Instruction>
You are an answer-judging agent.
Compare the predicted answer against the dataset ground truth answer and give a correctness score in [0, 1].
Rules:
- 1.0 means fully correct.
- 0.0 means completely wrong.
- If the prediction mentions key words/phrases from the ground truth answer, give partial credit (>0), not only 0/1.
- Consider semantic equivalence, aliases, and minor surface-form variation.
- You MUST output a decimal number in [0,1], e.g., 0.35, 0.70, 1.00.
- Output EXACTLY two lines in this format:
Score: <float between 0 and 1>
GlobalCritique: <one-sentence evaluation of final answer quality>
</Instruction>

<Question>
{question}
</Question>

<PredictedAnswer>
{predicted}
</PredictedAnswer>

<GroundTruthAnswer>
{ground_truth_answer}
</GroundTruthAnswer>
"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        if state_dicts and state_dicts[0].get("method", "").startswith("multiAgentGoT"):
            # Sequential multi-hop stores hop traces in hop_history on the *current* state.
            base = state_dicts[0]
            hist = base.get("hop_history") or []
            items = hist if isinstance(hist, list) and hist else state_dicts

            parts = []
            for s in items:
                if isinstance(s, dict) and "subquestion" in s and "partial_answer" in s:
                    subq = s.get("subquestion", "")
                    part = s.get("partial_answer") or ""
                    ev = s.get("evidence_summary", "")
                else:
                    subq = s.get("subquestion", "") if isinstance(s, dict) else ""
                    part = (s.get("partial_answer") or s.get("current", "")) if isinstance(s, dict) else ""
                    ev = s.get("evidence_summary", "") if isinstance(s, dict) else ""

                parts.append(f"- {subq}: {part}")
                if ev:
                    parts.append(f"  Evidence: {ev}")
            return self.ma_aggregate_prompt.format(
                question=base.get("question", ""),
                partials_text="\n".join(parts),
            )
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
                g0_start = 1
                g0_end = min(num_docs, half + 1)
                g1_start = max(1, half)
                g1_end = num_docs
                return self.got_split_prompt.format(
                    question=question,
                    context_text=context_text,
                    num_docs=num_docs,
                    g0_start=g0_start,
                    g0_end=g0_end,
                    g1_start=g1_start,
                    g1_end=g1_end,
                )
            if phase == 1:
                return self.got_partial_prompt.format(
                    question=question, current=kwargs.get("current", "")
                )
            return self.io_prompt.format(question=question, context_text=context_text)
        if method.startswith("multiAgentGoT"):
            role = kwargs.get("agent_role", "planner")
            if role == "planner":
                return self.ma_planner_prompt.format(
                    question=question,
                    context_text=context_text,
                    max_subquestions=kwargs.get("max_subquestions", 4),
                )
            if role == "retriever":
                bindings = kwargs.get("bindings") or {}
                if isinstance(bindings, dict) and bindings:
                    bindings_text = "\n".join([f"{k} = {v}" for k, v in bindings.items()])
                else:
                    bindings_text = ""
                return self.ma_retriever_prompt.format(
                    subquestion=kwargs.get("subquestion", ""),
                    context_text=context_text,
                    bindings_text=bindings_text,
                )
            if role == "reasoner":
                bindings = kwargs.get("bindings") or {}
                if isinstance(bindings, dict) and bindings:
                    bindings_text = "\n".join([f"{k} = {v}" for k, v in bindings.items()])
                else:
                    bindings_text = ""
                return self.ma_reasoner_prompt.format(
                    question=question,
                    subquestion=kwargs.get("subquestion", ""),
                    evidence_summary=kwargs.get("evidence_summary", ""),
                    bindings_text=bindings_text,
                )
            if role == "critic":
                return self.ma_critic_prompt.format(
                    question=question,
                    subquestion=kwargs.get("subquestion", ""),
                    partial_answer=kwargs.get("partial_answer", ""),
                    evidence_summary=kwargs.get("evidence_summary", ""),
                )
            return self.io_prompt.format(question=question, context_text=context_text)
        return self.io_prompt.format(question=question, context_text=context_text)

    def improve_prompt(self, **kwargs) -> str:
        return ""

    def validation_prompt(self, **kwargs) -> str:
        return ""

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        if not state_dicts:
            return ""
        state = state_dicts[0]
        method = state.get("method", "")
        if method.startswith("multiAgentGoT"):
            predicted = state.get("answer") or state.get("current") or state.get("partial_answer") or ""
            ground_truth_answer = state.get("ground_truth_answer", "")
            return self.ma_score_prompt.format(
                question=state.get("question", ""),
                predicted=predicted,
                ground_truth_answer=ground_truth_answer,
            )
        return ""
