"""端到端：单跳题走完五个节点，答案非空、selected 非空、trace 顺序固定。

"trace 顺序固定"不是洁癖：它是唯一能证明"图按预期拓扑跑"的观测点。少了 plan、
或者 decompose 出现在 retrieve 之后，都说明边接错了，而答案可能照样看起来正常。
"""

from __future__ import annotations

from app.graph import build_graph, run
from app.nodes import REFUSAL
from app.schemas import State

from tests.conftest import FakeLlm, FakeRetriever, FakeReranker, make_candidate


def test_happy_path_single_hop(cfg):
    llm = FakeLlm(route="complex", plan="generate", answer="CRAG contains 4,409 QA pairs [1].")
    retriever = FakeRetriever(batches=[[make_candidate("crag#009")]])
    out = run(cfg, llm, retriever, FakeReranker(), "How many QA pairs does CRAG contain?")

    assert out.answer and not out.answer.startswith(REFUSAL)
    assert [c.chunk_id for c in out.selected] == ["crag#009"]
    assert [s.node for s in out.trace] == ["route", "decompose", "retrieve", "plan", "answer"]
    assert all(s.ms >= 0 for s in out.trace)
    assert out.degraded == []


def test_answer_sees_numbered_context(cfg):
    """片段必须以 [1] 编号送进 prompt，否则"引用编号"这个要求无法被满足。"""
    llm = FakeLlm(route="complex", plan="generate")
    retriever = FakeRetriever(batches=[[make_candidate("crag#009"), make_candidate("crag#000")]])
    run(cfg, llm, retriever, FakeReranker(), "q")

    prompt = llm.seen[-1]
    assert "[1]" in prompt and "[2]" in prompt
    assert "crag" in prompt, "片段要带 source/heading，模型才有依据说'哪篇没写'"


def test_state_survives_roundtrip(cfg):
    """LangGraph 可能把 State 传成 dict；`run` 必须能还原成 State，否则下游访问属性会炸。"""
    llm = FakeLlm(route="complex", plan="generate")
    retriever = FakeRetriever(batches=[[make_candidate("crag#001")]])
    out = run(cfg, llm, retriever, FakeReranker(), "q")

    assert isinstance(out, State)
    assert State.model_validate(out.model_dump()).question == "q"


def test_simple_has_no_selected_but_still_answers(cfg):
    llm = FakeLlm(route="simple", answer="Retrieval-Augmented Generation.")
    out = run(cfg, llm, FakeRetriever(), FakeReranker(), "What does RAG stand for?")

    assert out.selected == []
    assert out.answer == "Retrieval-Augmented Generation."
    assert len(out.trace) == 2
