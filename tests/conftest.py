"""测试夹具：假模型、假召回、假重排。全部离线，秒级。

为什么不用 `monkeypatch` 打桩 SDK：`Llm` 已经留了 `client` 注入点，`make_nodes` 接受任意
`llm` 对象。按接口注入比按模块打桩可靠——打桩只验证"我调用了某个函数"，接口注入验证
"这个节点在给定输入下产出了什么状态"。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 测试不该依赖 .env：没有就在进程内补占位值（不会被真的调用）
for key, val in (("OPENAI_API_KEY", "test"), ("OPENAI_BASE_URL", "http://localhost/v1"), ("MODEL_NAME", "test-model")):
    os.environ.setdefault(key, val)

from app.config import Config, load_config  # noqa: E402
from app.schemas import Candidate, PlanOut, RouteOut, SubQuestionsOut  # noqa: E402


@pytest.fixture(scope="session")
def base_cfg() -> Config:
    return load_config(ROOT / "config.yaml")


@pytest.fixture
def cfg(base_cfg) -> Config:
    return base_cfg.model_copy(deep=True)


class FakeLlm:
    """按 prompt 首句分派节点。`answer` 固定引用 [1]，便于断言"生成确实看到了片段"。"""

    def __init__(self, route: str = "complex", plan: str = "generate", answer: str = "答案是 4,409 [1]."):
        self.route = route
        self.plan = plan
        self.answer = answer
        self.calls = 0
        self.generate_calls = 0
        self.tokens = 0
        self.seen: list[str] = []

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        self.generate_calls += 1
        self.seen.append(prompt)
        return self.answer

    def complete_structured(self, prompt: str, schema, system: str | None = None):
        self.calls += 1
        if "You decide whether" in prompt:
            return RouteOut(strategy=self.route, reason="fake route")
        if "Break the question" in prompt:
            return SubQuestionsOut(sub_questions=["first sub question", "second sub question"], reason="fake")
        if "Decide the next action" in prompt:
            return PlanOut(next_action=self.plan, reason="fake plan")
        raise AssertionError(f"未预期的 prompt：{prompt[:60]}")


class FakeRetriever:
    """按脚本逐次返回召回结果；`boom` 为真时抛 `RetrievalError`。"""

    def __init__(self, batches: list[list[Candidate]] | None = None, boom: bool = False):
        self.batches = list(batches or [])
        self.boom = boom
        self.queries: list[list[str]] = []

    def search(self, queries, k=None, hybrid=None, fusion=None):
        from app.retrieve import RetrievalError

        self.queries.append(list(queries))
        if self.boom:
            raise RetrievalError("两路召回均失败")
        batch = self.batches.pop(0) if self.batches else []
        return batch, []


class FakeReranker:
    def __init__(self, raise_error: bool = False):
        self.raise_error = raise_error
        self.calls = 0

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        from app.rerank import RerankError

        self.calls += 1
        if self.raise_error:
            raise RerankError("fake rerank failure")
        return candidates[:top_n]


def patch_cfg(cfg, graph: dict | None = None, retrieval: dict | None = None) -> Config:
    """配置是 frozen 的：测试要改变体必须走 model_copy，不能就地赋值（就地赋值本身也该报错）。"""
    return cfg.model_copy(
        update={
            "graph": cfg.graph.model_copy(update=graph or {}),
            "retrieval": cfg.retrieval.model_copy(update=retrieval or {}),
        }
    )


def make_candidate(cid: str, source: str | None = None, score: float = 0.5) -> Candidate:
    return Candidate(
        chunk_id=cid,
        source=source or cid.split("#")[0],
        heading=f"heading-{cid}",
        text=f"text of {cid}",
        vector_score=score,
        bm25_score=None,
        rrf=score,
    )


@pytest.fixture
def fake_llm() -> FakeLlm:
    return FakeLlm()
