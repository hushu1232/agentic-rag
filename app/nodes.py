"""五个节点。每个节点只做一件事，且只返回自己负责的字段（ADR-011 的消费关系）。

三条纪律：
1. **代码层熔断优先于模型**：`plan` 之前先查轮数与子问题是否耗尽，模型没有机会说"再来一轮"。
   源实现的多跳回边从来没执行过（`rag-multihop.mjs:234` 读 `state.strategy`），根因就是
   "条件边读的字段"和"节点写的字段"不是同一个——这里条件边只读 `plan` 写的 `next_action`。
2. **空上下文不进生成**：召回为空时直接产出拒答，生成调用次数必须为 0（可断言）。
3. **拒答走固定标记**：`I cannot confirm this from the provided sources` 作为首句，规则可判。
"""

from __future__ import annotations

from typing import Any

from app.llm import LlmProto
from app.rerank import RerankError
from app.schemas import Candidate, NextAction, PlanOut, RouteOut, State, SubQuestionsOut, TraceStep

REFUSAL = "I cannot confirm this from the provided sources"

ROUTE_PROMPT = """You decide whether answering the question requires looking up specific facts from a \
fixed corpus of 14 RAG research papers (NeurIPS/ICLR/ACL/EMNLP/NAACL/TACL).

Choose "simple" ONLY when the question needs no corpus fact: arithmetic, a definition that is general \
knowledge, a greeting, or a question about the user's own setup.
Choose "complex" when it asks for a number, a reported result, a method detail, a benchmark property, or \
requires combining two facts.

Reply with JSON only: {"strategy": "simple" | "complex", "reason": "<one sentence>"}"""

DECOMPOSE_PROMPT = """Break the question into 1-6 ordered English sub-questions that can each be used as a \
standalone search query against a corpus of RAG papers.

Rules:
- Every sub-question must name its entities in full. Never use pronouns or "this method" / "that paper".
- Order them so that earlier ones establish the entity and later ones ask about its properties.
- Do not just copy the original question as the only item unless it truly cannot be split.

Reply with JSON only: {"sub_questions": ["...", "..."], "reason": "<one sentence>"}"""

PLAN_PROMPT = """Decide the next action for a multi-hop retrieval loop.

You are given the question, the ordered sub-questions (with the ones already searched marked DONE), and a \
short list of the headings of what has been retrieved so far.

Choose "retrieve" if a sub-question is still unsearched and it would add new evidence.
Choose "generate" if the evidence is already enough, or if searching more cannot help.

Reply with JSON only: {"next_action": "retrieve" | "generate", "reason": "<one sentence>"}"""

ANSWER_PROMPT = f"""Answer the question using ONLY the numbered context passages below.

Rules:
- Cite the passage number in brackets, e.g. [2], for every factual claim.
- Do not use knowledge that is not in the passages. Do not guess numbers.
- If the passages do not settle a key fact, your answer must start with exactly this sentence:
  "{REFUSAL}"
  then name the single missing fact in one sentence.

Context:
{{context}}

Question: {{question}}"""


def _refusal(missing: str) -> str:
    return f"{REFUSAL} The corpus does not state {missing}."


def _context(selected: list[Candidate]) -> str:
    return "\n\n".join(f"[{i}] {c.source} › {c.heading}\n{c.text}" for i, c in enumerate(selected, 1))


def _trace(state: State, node: str, ms: int, decision: str, detail: str = "") -> list[TraceStep]:
    return [*state.trace, TraceStep(node=node, ms=ms, decision=decision, detail=detail)]


def make_nodes(cfg, llm: LlmProto, retriever, reranker):
    """返回五个可挂到图上的节点函数。依赖全部从闭包进来，测试时可整体替换。"""

    def route(state: State) -> dict[str, Any]:
        import time

        t0 = time.perf_counter()
        out: RouteOut = llm.complete_structured(ROUTE_PROMPT, RouteOut, state.question)
        return {
            "strategy": out.strategy,
            "route_reason": out.reason,
            "trace": _trace(state, "route", int((time.perf_counter() - t0) * 1000), out.strategy.value, out.reason),
        }

    def decompose(state: State) -> dict[str, Any]:
        import time

        t0 = time.perf_counter()
        degraded: list[str] = []
        try:
            out: SubQuestionsOut = llm.complete_structured(DECOMPOSE_PROMPT, SubQuestionsOut, state.question)
            subs = [s.strip() for s in out.sub_questions if s.strip()][: cfg.graph.max_sub_questions]
            reason = out.reason
        except Exception as exc:  # 拆解失败不该让整次运行失败，退化为单条
            subs, reason = [], f"decompose failed: {type(exc).__name__}"
        if not subs:
            subs = [state.question]
            degraded = ["decompose_degraded"]
        return {
            "sub_questions": subs,
            "next_idx": 0,
            "degraded": state.degraded + degraded,
            "trace": _trace(state, "decompose", int((time.perf_counter() - t0) * 1000), f"{len(subs)} 条", reason),
        }

    def retrieve(state: State) -> dict[str, Any]:
        import time

        t0 = time.perf_counter()
        idx = min(state.next_idx, len(state.sub_questions) - 1) if state.sub_questions else 0
        query = state.sub_questions[idx] if state.sub_questions else state.question
        got, degraded_now = retriever.search([query])

        seen = {c.chunk_id for c in state.candidates}
        merged = state.candidates + [c for c in got if c.chunk_id not in seen]
        return {
            "current_query": query,
            "next_idx": idx + 1,
            "rounds": state.rounds + 1,
            "candidates": merged,
            "fused": got,
            "degraded": sorted(set(state.degraded + degraded_now)),
            "trace": _trace(
                state, "retrieve", int((time.perf_counter() - t0) * 1000), query, f"召回 {len(got)} 条，累计 {len(merged)} 条"
            ),
        }

    def plan(state: State) -> dict[str, Any]:
        import time

        t0 = time.perf_counter()
        # 两道硬熔断先于模型：这里 return 的值就是条件边的唯一依据。
        if cfg.graph.max_rounds <= state.rounds:
            action, why = "generate", f"熔断：轮数已达上限 {cfg.graph.max_rounds}"
        elif state.next_idx >= len(state.sub_questions):
            action, why = "generate", "熔断：子问题已用尽"
        elif not state.candidates:
            action, why = "generate", "熔断：召回为空，继续检索无意义"
        else:
            done = "; ".join(
                f"{i + 1}. {'DONE' if i < state.next_idx else 'TODO'} {q}" for i, q in enumerate(state.sub_questions)
            )
            heads = "; ".join(f"{c.source}›{c.heading}" for c in state.fused[:5])
            brief = f"Question: {state.question}\nSub-questions: {done}\nRetrieved headings: {heads}"
            try:
                out: PlanOut = llm.complete_structured(PLAN_PROMPT, PlanOut, brief)
                action, why = out.next_action.value, out.reason
            except Exception as exc:
                action, why = "generate", f"plan failed: {type(exc).__name__}"
            # 子问题还有剩余时才允许继续；模型说 retrieve 但没得检索，按 generate 处理。
            if action == "retrieve" and state.next_idx >= len(state.sub_questions):
                action, why = "generate", "模型要求继续，但无剩余子问题"

        return {
            "next_action": NextAction(action),
            "trace": _trace(state, "plan", int((time.perf_counter() - t0) * 1000), action, why),
        }

    def answer(state: State) -> dict[str, Any]:
        import time

        t0 = time.perf_counter()
        # 只有「本该检索却什么都没召回」才算拒答。simple 分支本来就不检索，
        # 若也在这里拦下，route 判 simple 就永远只能回答"无法从资料确认"——
        # 这条是 T14 的 test_simple_has_no_selected_but_still_answers 抓出来的。
        if state.strategy != "simple" and not state.fused and not state.candidates:
            missing = state.current_query or "the requested fact"
            return {
                "answer": _refusal(missing),
                "selected": [],
                "trace": _trace(state, "answer", int((time.perf_counter() - t0) * 1000), "refused", "召回为空"),
            }

        try:
            selected = reranker.rerank(state.current_query or state.question, state.fused, cfg.retrieval.top_n)
            degraded_now: list[str] = []
        except RerankError:
            selected, degraded_now = state.fused[: cfg.retrieval.top_n], ["rerank_degraded"]

        ctx = _context(selected)
        prompt = ANSWER_PROMPT.replace("{context}", ctx).replace("{question}", state.question)
        text = llm.complete(prompt).strip()
        return {
            "answer": text,
            "selected": selected,
            "context": ctx,
            "degraded": sorted(set(state.degraded + degraded_now)),
            "trace": _trace(state, "answer", int((time.perf_counter() - t0) * 1000), f"{len(selected)} 片段", ""),
        }

    return route, decompose, retrieve, plan, answer
