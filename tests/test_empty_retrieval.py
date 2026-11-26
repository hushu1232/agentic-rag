"""D03 / D05 / D06：空召回必须拒答，且生成调用次数为 0。

源实现里三处叠加：`if (!retrievedDocs)` 真值写反、`if (!retrieveDocs.length)` 拿函数取
`.length`（undefined，恒为假）、以及判空写在生成之后。三者任一没修，空召回都会走到
"先编一段，再说没找到"。所以断言必须是**生成调用次数 == 0**，而不是"答案里有拒答词"。
"""

from __future__ import annotations

import pytest

from app.graph import build_graph
from app.nodes import REFUSAL
from app.schemas import State

from tests.conftest import FakeLlm, FakeRetriever, FakeReranker, make_candidate, patch_cfg


def _run(cfg, retriever, reranker=None):
    llm = FakeLlm(route="complex", plan="generate")
    graph = build_graph(cfg, llm, retriever, reranker or FakeReranker())
    out = graph.invoke(State(question="Why do refunds take 5 days?", max_rounds=2),
                       config={"recursion_limit": 20})
    return out, llm


def test_empty_retrieval_refuses_without_generating(cfg):
    out, llm = _run(cfg, FakeRetriever(batches=[[]]))

    assert llm.generate_calls == 0, "召回为空时不许调用生成模型"
    assert out["answer"].startswith(REFUSAL)
    assert out["selected"] == []
    assert out["trace"][-1].decision == "refused"


def test_refusal_names_the_missing_fact(cfg):
    out, _ = _run(cfg, FakeRetriever(batches=[[]]))

    assert "does not state" in out["answer"], "拒答要说明缺哪一项，不能只说'不知道'"
    assert out["current_query"], "拒答要能追溯到最近一轮检索的查询"


def test_rerank_failure_degrades_instead_of_failing(cfg):
    """重排挂了不该让整次运行失败：降级为不重排，并在 degraded 里留痕。"""
    out, llm = _run(cfg, FakeRetriever(batches=[[make_candidate("crag#001"), make_candidate("crag#002")]]),
                    reranker=FakeReranker(raise_error=True))

    assert "rerank_degraded" in out["degraded"]
    assert llm.generate_calls == 1
    assert len(out["selected"]) == 2


def test_selected_is_capped_by_top_n(cfg):
    cfg = patch_cfg(cfg, retrieval={"top_n": 3})
    batch = [make_candidate(f"crag#{i:03d}") for i in range(10)]
    out, _ = _run(cfg, FakeRetriever(batches=[batch]))

    assert len(out["selected"]) == 3, "送进生成的片段数必须受 top_n 约束"


def test_retrieval_error_propagates(cfg):
    """两路都失败不是"空召回"，必须抛出来——否则会被当成'资料里没有'而拒答。"""
    from app.retrieve import RetrievalError

    with pytest.raises(RetrievalError):
        _run(cfg, FakeRetriever(boom=True))
