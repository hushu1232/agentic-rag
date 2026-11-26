"""D02 / 路由：simple 不许进检索。

这条测试真正拦的是源实现里那类"字段写错了但静默跑通"的问题：如果 `route` 把策略写进
别的字段，条件边读不到，图会默认走向 retrieve——这里用"检索调用次数"把它钉死。
"""

from __future__ import annotations

from app.graph import build_graph
from app.schemas import State

from tests.conftest import FakeLlm, FakeRetriever, FakeReranker, make_candidate


def _run(cfg, llm, retriever):
    graph = build_graph(cfg, llm, retriever, FakeReranker())
    return graph.invoke(State(question="What does RAG stand for?", max_rounds=cfg.graph.max_rounds),
                        config={"recursion_limit": 20})


def test_simple_never_retrieves(cfg):
    llm, retriever = FakeLlm(route="simple"), FakeRetriever()
    out = _run(cfg, llm, retriever)

    assert retriever.queries == [], "simple 分支不该发起任何检索"
    assert [s.node for s in out["trace"]] == ["route", "answer"]
    assert out["selected"] == []
    assert llm.generate_calls == 1, "只有 answer 一次生成"


def test_complex_goes_through_decompose(cfg):
    llm = FakeLlm(route="complex", plan="generate")
    retriever = FakeRetriever(batches=[[make_candidate("crag#009")]])
    out = _run(cfg, llm, retriever)

    nodes = [s.node for s in out["trace"]]
    assert nodes[0] == "route" and nodes[1] == "decompose", nodes
    assert len(retriever.queries) == 1
    assert out["strategy"] == "complex"
    assert out["sub_questions"] == ["first sub question", "second sub question"]


def test_strategy_field_is_what_the_edge_reads(cfg):
    """把 simple 的返回值改成非法值，pydantic 必须当场拒绝（而不是静默当 complex）。"""
    import pytest
    from pydantic import ValidationError

    from app.schemas import RouteOut

    with pytest.raises(ValidationError):
        RouteOut(strategy="simple ", reason="尾部空格")
