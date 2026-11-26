"""D01 / 熔断：回边必须真的回到 retrieve，且次数可控。

源实现的 `afterPlan` 读 `state.strategy`（route 阶段写入、恒为 complex），所以
`retrieve → plan → retrieve` 一次都没执行过，而日志里 `plannedNext` 打得漂漂亮亮。
这条测试的断言只有一个数字：**检索被调用了几次**。把条件边改回去读 `strategy`，
这个数字立刻从 2 掉到 1。
"""

from __future__ import annotations

from app.graph import build_graph
from app.schemas import State

from tests.conftest import FakeLlm, FakeRetriever, FakeReranker, make_candidate, patch_cfg


def _run(cfg, plan: str, batches: int = 5):
    llm = FakeLlm(route="complex", plan=plan)
    retriever = FakeRetriever(batches=[[make_candidate(f"crag#{i:03d}")] for i in range(batches)])
    graph = build_graph(cfg, llm, retriever, FakeReranker())
    out = graph.invoke(State(question="multi hop question", max_rounds=cfg.graph.max_rounds),
                       config={"recursion_limit": 4 * cfg.graph.max_rounds + 6})
    return out, retriever


def test_back_edge_actually_loops(cfg):
    cfg = patch_cfg(cfg, graph={"max_rounds": 2})
    out, retriever = _run(cfg, plan="retrieve")

    assert len(retriever.queries) == 2, f"max_rounds=2 应恰好检索 2 次，实际 {len(retriever.queries)}"
    assert out["rounds"] == 2
    assert out["trace"][-1].node == "answer"
    assert [s.node for s in out["trace"]].count("retrieve") == 2
    assert [s.node for s in out["trace"]].count("plan") == 2


def test_max_rounds_one_stops_immediately(cfg):
    cfg = patch_cfg(cfg, graph={"max_rounds": 1})
    out, retriever = _run(cfg, plan="retrieve")

    assert len(retriever.queries) == 1, "熔断必须在 plan 之后、retrieve 之前生效"
    assert out["trace"][-1].node == "answer"


def test_generate_ends_after_single_round(cfg):
    out, retriever = _run(cfg, plan="generate")

    assert len(retriever.queries) == 1
    assert [s.node for s in out["trace"]] == ["route", "decompose", "retrieve", "plan", "answer"]


def test_exhausted_subquestions_forces_generate(cfg):
    """plan 说 retrieve、但子问题已用尽时必须改判 generate，否则图会空转。"""
    cfg = patch_cfg(cfg, graph={"max_rounds": 5})
    out, retriever = _run(cfg, plan="retrieve", batches=6)
    # 2 条子问题 → 最多检索 2 次，第 3 次之前被 next_idx 熔断
    assert len(retriever.queries) == 2
    assert out["next_action"] == "generate"


def test_multihop_off_removes_plan_node(cfg):
    cfg = patch_cfg(cfg, graph={"multihop": False})
    llm = FakeLlm(route="complex", plan="retrieve")
    retriever = FakeRetriever(batches=[[make_candidate("crag#001")]])
    graph = build_graph(cfg, llm, retriever, FakeReranker())
    out = graph.invoke(State(question="q", max_rounds=3), config={"recursion_limit": 20})

    assert "plan" not in [s.node for s in out["trace"]]
    assert len(retriever.queries) == 1, "单跳模式下不该有回边"
