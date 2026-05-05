from typing import Dict, List

from graph_of_thoughts import prompter

try:
    from multi_hop_qa.utils import gotContextExcerpt, gotEvidenceTopSentences
except ImportError:
    import utils as _mh_utils

    gotContextExcerpt = _mh_utils.gotContextExcerpt
    gotEvidenceTopSentences = _mh_utils.gotEvidenceTopSentences


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
The summary below is a distilled view of ONE document-range; it may miss names or bridging facts.

Use BOTH:
1) Top evidence lines retrieved for this question (+summary), AND
2) The full-context excerpt,

to derive the key partial fact aligned with grounded wording. Prefer facts supported by Evidence; treat Summary only as hints. Be concise. End with "Partial: ...".

If the fact is a definite single value (number, name, yes/no, etc.), write only that value after "Partial:" (no full sentence). Otherwise one short sentence is OK.
</Instruction>

<EvidenceTopSentences>
{evidence_topk}
</EvidenceTopSentences>

<FullContextExcerpt>
{context_excerpt}
</FullContextExcerpt>

<Summary>
{current}
</Summary>

<Question>
{question}
</Question>

Partial:"""

    got_aggregate_prompt = """<Instruction>
Combine partial answers below into the correct final answer, using Evidence + Context excerpt when partials disagree or omit detail.
Output exactly one line: "Answer: <final answer>".

If the answer is definite and can be expressed as a single token or minimal phrase (number, name, yes/no, nationality, etc.), output ONLY that after "Answer:"—no full sentence.
</Instruction>

<Question>
{question}
</Question>

<EvidenceTopSentences>
{evidence_topk}
</EvidenceTopSentences>

<FullContextExcerpt>
{context_excerpt}
</FullContextExcerpt>

Partial answer 1: {input1}

Partial answer 2: {input2}

Answer:"""

    ma_planner_prompt = """<Instruction>
You are the Planner agent in a 4-agent multi-hop QA system.
Decompose the question into at most {max_subquestions} executable sub-questions.
Each sub-question should introduce ONE new relation or attribute needed for the next step
(e.g. identify entity → where it is / when / what it owns → name of a part/place tied to that).
Avoid repeating the same relation in consecutive steps; later steps must depend on earlier answers.
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
- Temporal semantics hint (IMPORTANT):
  - If the SubQuestion asks "When/What year/Date" about an EVENT (e.g., "impeached", "founded", "announced", "launched"),
    prioritize evidence that states the EVENT OCCURRED/STARTED in a specific year/date (e.g., "in 1786", "on 7 January 2011").
  - Do NOT prefer evidence that only gives the DURATION of a later process (e.g., "between 1788 and 1795") unless the question explicitly asks for a duration/range ("between/from-to/how long").
- Employer vs self-employment (IMPORTANT):
  - If the SubQuestion asks who "employs", "hired", or is the "employer" of a person, prefer evidence that states an employment or employer relationship with another organization/studio.
  - Do NOT answer with that person's own company, personal studio, or "operates/owns" business unless the SubQuestion explicitly asks for their own venture.
- Multi-hop evidence shift (IMPORTANT):
  - Each SubQuestion may require a DIFFERENT passage than the previous hop. Do NOT keep citing the same title/sentence index by default if that span does not state the relation asked NOW
    (e.g. "headquarters", "based in", "located in", "capital of", "born in", "castle in", "river in").
  - When <Bindings> names an entity from a prior hop, actively search the Context for OTHER documents that connect that entity to the relation in the SubQuestion (headquarters, city, country, building, etc.).
  - You MAY return multiple evidence_spans from different document titles if needed to bridge entity → place → attribute.
- Binding co-occurrence (CRITICAL):
  - If <Bindings> is non-empty, every cited sentence MUST mention the bound value(s) literally (e.g. #1's organization name) OR clearly refer to that same entity in the same sentence.
  - Do NOT pick a paragraph only because it contains the generic word "headquarters" for a different organization (e.g. Apple, Yale, a random NGO) with no mention of #1.
  - For "castle/building at the headquarters" of a bound company, prefer passages that link THAT company or its known location (#2) to a named building/castle; avoid unrelated associations whose only match is "headquarters" + "castle".

Return ONLY JSON in this exact format:
{{"evidence_spans":[["title", sent_idx], ...], "evidence_summary":"one short evidence summary grounded in the cited span(s)"}}
</Instruction>

<RetryFeedback>
{retry_feedback_text}
</RetryFeedback>

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
Grounding constraints (CRITICAL):
- Do NOT speculate or add any detail not explicitly supported by <Evidence>.
- Do NOT introduce new entities, dates, years, numbers, or ranges that do not appear in <Evidence>.
- If <Evidence> mentions only a single year/date, do NOT invent an additional year/date.
- If <SubQuestion> asks who employs / is the employer of someone, answer the hiring organization; if <Evidence> mentions both an employer and that person's own studio/company, choose the employer unless the question asks for the personal business.
- Answer-type alignment (IMPORTANT):
  - If <SubQuestion> asks WHERE / headquarters / location / city / country / "based in", the Partial MUST be a geographic answer (place name(s)) explicitly supported by <Evidence>. Do NOT output only an organization or person name unless <Evidence> states that place.
  - If <SubQuestion> asks for a building/landmark/castle/river/mountain/etc., the Partial MUST name that kind of entity from <Evidence>, not a parent company or unrelated entity.
  - If <Evidence> does not support the asked relation, output exactly: Partial: NEED_RETRIEVE (do not output a wrong-type guess or a long explanation instead).
- NEED_RETRIEVE discipline (IMPORTANT):
  - If <Evidence> already contains a short entity (organization/person/place) that directly answers <SubQuestion>, output it as Partial — do NOT use NEED_RETRIEVE.
  - Example: if the question asks for the employer company and <Evidence> says the person directed "CompanyX's 1985 film", then CompanyX is the supported employer answer.

Return exactly:
Partial: <partial answer>
Confidence: <0.0-1.0>
</Instruction>

<Question>
{question}
</Question>

<RetryFeedback>
{retry_feedback_text}
</RetryFeedback>

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
TimeFacet: <one of: initiation | occurrence | trial_duration | duration | other>
ReasonCode: <one of: wrong_entity, insufficient_evidence, not_grounded, other>
SuggestedAction: <one of: backtrack_retrieve, backtrack_reason, accept>
Validation: <PASS or REJECT>
RefinedPartial: <better partial answer>
Confidence: <0.0-1.0>

Guidelines for TimeFacet:
- Use "initiation" when the question asks when an action/event was initiated (e.g., "impeached", "charged", "founded").
- Use "occurrence" when the question asks when something happened at a point in time (unveiled, released, born, died).
- Use "trial_duration"/"duration" when the evidence/answer is a span/range (between X and Y, from X to Y, lasted N years).
- IMPORTANT benchmark convention:
  - For this multi-hop QA benchmark, questions phrased as "When was X impeached?" are commonly answered by the YEAR the impeachment
    TRIAL/PROCEEDINGS began (a single year), if that is what the evidence directly states (e.g., "trial during 1786").
  - If your evidence provides ONLY a RANGE ("between 1788 and 1795") and the question is NOT asking for a duration/range,
    you MUST set TimeFacet=trial_duration and Validation=REJECT with SuggestedAction=backtrack_retrieve (need a single year/date).
- Employer disambiguation:
  - If the SubQuestion is about who employs someone and <Evidence> supports a clear employer distinct from that person's own studio/company, but <PartialAnswer> names only the personal studio, set ReasonCode=wrong_entity, SuggestedAction=backtrack_retrieve, Validation=REJECT.
- Employer / studio from possessive phrasing:
  - If <SubQuestion> asks for the company that employs or hired the person and <Evidence> states they directed/produced/wrote for a named company's film/show/project (e.g. "Company's 1985 film"), treat that company name as the supported employer: Validation=PASS and RefinedPartial=that company name (from <Evidence> only), unless the SubQuestion explicitly asks for their personal studio instead.
- Castle / headquarters relevance:
  - If <SubQuestion> asks for a castle or building at the headquarters and <Bindings> includes #1 (company) or #2 (place), but <Evidence> does not mention #1 or #2 and instead describes another organization's headquarters, set Validation=REJECT, ReasonCode=wrong_entity, SuggestedAction=backtrack_retrieve.
- Answer-type vs subquestion (IMPORTANT):
  - If the SubQuestion asks for a place (where/headquarters/location/city/country) but <Evidence> contains no supported place and <PartialAnswer> is only an org/person name or a long non-place reply, set Validation=REJECT, ReasonCode=insufficient_evidence, SuggestedAction=backtrack_retrieve.
  - If the SubQuestion asks for a specific kind of entity (castle, building, river, date, etc.) but <PartialAnswer> is a different kind (e.g. company name for a "castle" question), set Validation=REJECT with ReasonCode=wrong_entity unless <Evidence> explicitly equates them.
  - When evidence truly lacks the requested facet, set RefinedPartial to exactly: NEED_RETRIEVE (not a paragraph). Validation may be PASS only if PartialAnswer is already NEED_RETRIEVE and evidence is correctly empty for that facet.

Grounding constraints (CRITICAL):
- RefinedPartial MUST be fully supported by <Evidence>.
- If PartialAnswer contains extra facts not in <Evidence>, remove them in RefinedPartial.
- Do NOT add any new entities/dates/years/numbers that do not appear in <Evidence>.
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
- Final-question alignment (CRITICAL):
  - First infer what the <Question> asks for: person, organization, place/city/country, building/landmark, date/time, number, yes/no, etc.
  - Your final answer MUST be that type. Do NOT substitute an intermediate-hop answer of a different type
    (e.g. if the question asks for a castle or city, do NOT answer with a company name from an earlier hop).
  - Use the hop that corresponds to the FINAL facet asked by the <Question>; earlier hops are only supporting context.
  - If no partial or evidence supports the requested facet, output: Answer: Not mentioned
- Temporal precision rule (IMPORTANT): if the question is asking "when"/a date/time AND the provided partials/evidence contain a more specific date than just a year, you MUST keep the finest supported granularity.
  Examples:
  - If you see "April 2012" anywhere relevant, do NOT answer only "2012"; answer "April 2012".
  - If you see "7 January 2011", do NOT answer "2011"; answer "7 January 2011".
  - Only output a bare year (e.g., "2012") if month/day are NOT present in the provided partials/evidence.
- Anti-hallucination rule (IMPORTANT): for numeric/time answers, output ONLY values that appear verbatim in the provided partials/evidence summaries.
- Reliability rule (IMPORTANT): each clue includes a line_score in [0,1], where higher is more reliable.
  You MUST prioritize clues with higher line_score when clues conflict.
  If a clue has max_retry_reached=true and low line_score, treat it as weak evidence.
  If a clue has validation=REJECT, it means this partial answer was explicitly rejected by the critic. You MUST ignore REJECTED answers whenever possible, and rely on the PASS answers instead.
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
                    line_score = s.get("line_score", s.get("confidence", 0.0))
                    trust = s.get("line_trust", "")
                    max_retry = s.get("max_retry_reached", False)
                    validation = s.get("validation_decision", "UNKNOWN")
                else:
                    subq = s.get("subquestion", "") if isinstance(s, dict) else ""
                    part = (s.get("partial_answer") or s.get("current", "")) if isinstance(s, dict) else ""
                    ev = s.get("evidence_summary", "") if isinstance(s, dict) else ""
                    line_score = s.get("line_score", s.get("confidence", 0.0)) if isinstance(s, dict) else 0.0
                    trust = s.get("line_trust", "") if isinstance(s, dict) else ""
                    max_retry = s.get("max_retry_reached", False) if isinstance(s, dict) else False
                    validation = s.get("validation_decision", "UNKNOWN") if isinstance(s, dict) else "UNKNOWN"

                parts.append(f"- {subq}: {part}")
                if ev:
                    parts.append(f"  Evidence: {ev}")
                parts.append(
                    f"  Reliability: validation={validation}, line_score={float(line_score or 0.0):.2f}, "
                    f"trust={trust or 'unknown'}, max_retry_reached={bool(max_retry)}"
                )
            return self.ma_aggregate_prompt.format(
                question=base.get("question", ""),
                partials_text="\n".join(parts),
            )
        assert len(state_dicts) == 2
        base = state_dicts[0]
        ctx = base.get("context") or []
        ct_full = str(base.get("context_text") or "")
        iq = "{}\n{}\n{}".format(
            base.get("question", ""),
            state_dicts[0].get("current", ""),
            state_dicts[1].get("current", ""),
        )
        agg_ev = gotEvidenceTopSentences(ctx, iq, top_k=10)
        agg_excerpt = gotContextExcerpt(ct_full, max_chars=5600)
        return self.got_aggregate_prompt.format(
            question=base.get("question", ""),
            input1=state_dicts[0].get("current", ""),
            input2=state_dicts[1].get("current", ""),
            evidence_topk=agg_ev,
            context_excerpt=agg_excerpt,
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
                ctx_list = kwargs.get("context") or []
                cq = "{}\n{}".format(question, kwargs.get("current", "") or "")
                partial_ev = gotEvidenceTopSentences(ctx_list, cq, top_k=8)
                partial_excerpt = gotContextExcerpt(
                    kwargs.get("context_text") or "", max_chars=5200
                )
                return self.got_partial_prompt.format(
                    question=question,
                    current=kwargs.get("current", ""),
                    evidence_topk=partial_ev,
                    context_excerpt=partial_excerpt,
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
                rf = kwargs.get("retry_feedback") or {}
                if isinstance(rf, dict) and rf:
                    retry_feedback_text = "\n".join([f"{k}: {v}" for k, v in rf.items() if v])
                else:
                    retry_feedback_text = ""
                return self.ma_retriever_prompt.format(
                    subquestion=kwargs.get("subquestion", ""),
                    context_text=context_text,
                    bindings_text=bindings_text,
                    retry_feedback_text=retry_feedback_text,
                )
            if role == "reasoner":
                bindings = kwargs.get("bindings") or {}
                if isinstance(bindings, dict) and bindings:
                    bindings_text = "\n".join([f"{k} = {v}" for k, v in bindings.items()])
                else:
                    bindings_text = ""
                rf = kwargs.get("retry_feedback") or {}
                if isinstance(rf, dict) and rf:
                    retry_feedback_text = "\n".join([f"{k}: {v}" for k, v in rf.items() if v])
                else:
                    retry_feedback_text = ""
                return self.ma_reasoner_prompt.format(
                    question=question,
                    subquestion=kwargs.get("subquestion", ""),
                    evidence_summary=kwargs.get("evidence_summary", ""),
                    bindings_text=bindings_text,
                    retry_feedback_text=retry_feedback_text,
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
