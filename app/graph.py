"""状态图。三条条件边，一条回边。

回边长这样：`retrieve → plan → (retrieve | answer)`。条件边**只读 `state.next_action`**，
而 `next_action` 只有 `plan` 会写——这是与源实现的关键差别：源实现的条件边读
`state.strategy`（route 阶段写入、值恒为 `complex`），于是回边永远不触发，而日志里
`plannedNext` 打印得好好的，谁也不会怀疑（`rag-multihop.mjs:228-236`）。

`multihop=false` 时不保留不可达节点：图编译成 `route → retrieve → answer`，`plan` 与
`decompose` 根本不挂上去（ADR-012 的风险处理——不留未测试分支）。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes import make_nodes
from app.schemas import State


def build_graph(cfg, llm, retriever, reranker):
    route, decompose, retrieve, plan, answer = make_nodes(cfg, llm, retriever, reranker)
    g = StateGraph(State)

    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)

    g.add_edge(START, "route")
    g.add_edge("answer", END)

    if not cfg.graph.multihop:
        g.add_conditional_edges("route", lambda s: "retrieve" if s.strategy == "complex" else "answer",
                               {"retrieve": "retrieve", "answer": "answer"})
        g.add_edge("retrieve", "answer")
        return g.compile()

    g.add_node("decompose", decompose)
    g.add_node("plan", plan)

    # simple 不检索：直接 answer，trace 只有两条（B1）
    g.add_conditional_edges("route", lambda s: "decompose" if s.strategy == "complex" else "answer",
                            {"decompose": "decompose", "answer": "answer"})
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "plan")
    # 回边：读 plan 写的字段，不读 route 写的字段
    g.add_conditional_edges("plan", lambda s: s.next_action.value if s.next_action else "generate",
                            {"retrieve": "retrieve", "generate": "answer"})
    return g.compile()


def run(cfg, llm, retriever, reranker, question: str) -> State:
    """一次独立运行。`recursion_limit` 按最坏路径给足：route+decompose+3×(retrieve+plan)+answer。"""
    graph = build_graph(cfg, llm, retriever, reranker)
    init = State(question=question, max_rounds=cfg.graph.max_rounds)
    out = graph.invoke(init, config={"recursion_limit": 4 * cfg.graph.max_rounds + 6})
    return out if isinstance(out, State) else State.model_validate(out)
