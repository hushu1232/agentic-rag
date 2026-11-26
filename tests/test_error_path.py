"""D04 / D10 / D11 / D14 / D15：错误必须被包装成已知类型，退出码必须非零。

源实现的三种收尾写法都通向同一个结局——打印一行日志、退出码仍是 0：
`.catch(console.error)`（传函数引用，拿不到错误对象）、`.then(err => ...)`（把结果当错误、
打印 undefined）、裸 `main()`（未捕获的 rejection 在 Node 里也常常不改变退出码）。
所以这里断言的是**类型**和**退出码**，不是"有没有打印"。
"""

from __future__ import annotations

import pytest

from app.cli import EXIT
from app.llm import Llm, LlmError
from app.retrieve import RetrievalError, Retriever

from tests.conftest import make_candidate


def test_single_route_failure_degrades(cfg, monkeypatch):
    """向量路挂了，关键词路还能用 → 降级而不是失败（D15）。"""
    import app.retrieve as rv

    def boom(*a, **k):
        raise RuntimeError("vector backend down")

    monkeypatch.setattr(rv, "vector_search", boom)
    monkeypatch.setattr(rv, "bm25_search", lambda *a, **k: [make_candidate("crag#009")])

    r = Retriever([], __import__("numpy").zeros((0, 4)), None, None, cfg)
    got, degraded = r.search(["q"])

    assert [c.chunk_id for c in got] == ["crag#009"]
    assert "retrieval_degraded:vector" in degraded


def test_both_routes_fail_raises_retrieval_error(cfg, monkeypatch):
    import app.retrieve as rv

    monkeypatch.setattr(rv, "vector_search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vec")))
    monkeypatch.setattr(rv, "bm25_search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bm25")))

    r = Retriever([], __import__("numpy").zeros((0, 4)), None, None, cfg)
    with pytest.raises(RetrievalError) as ei:
        r.search(["q"])

    assert ei.value.__cause__ is not None, "原始异常必须留在 __cause__，否则排查时只剩一句包装话"


def test_llm_error_wraps_and_keeps_cause(cfg):
    class Stub:
        def __init__(self):
            from types import SimpleNamespace as NS

            self.chat = NS(completions=NS(create=self._create))

        def _create(self, **kw):
            raise RuntimeError("connection reset by peer")

    llm = Llm(cfg, client=Stub())
    with pytest.raises(LlmError) as ei:
        llm.complete("hi")

    assert isinstance(ei.value.__cause__, RuntimeError)
    assert "connection reset" in str(ei.value), "原始信息必须透出，不能只剩'调用失败'"


def test_bad_json_raises_llm_error_not_jsondecode(cfg):
    from app.schemas import RouteOut

    class Stub:
        def __init__(self):
            from types import SimpleNamespace as NS

            self.chat = NS(completions=NS(create=self._create))

        def _create(self, **kw):
            from types import SimpleNamespace as NS

            return NS(choices=[NS(message=NS(content="<html>500</html>"))], usage=NS(total_tokens=1))

    llm = Llm(cfg, client=Stub())
    with pytest.raises(LlmError):
        llm.complete_structured("x", RouteOut)


def test_exit_codes_are_distinct_and_nonzero():
    assert EXIT["config"] != 0 and EXIT["llm"] != 0 and EXIT["retrieval"] != 0
    assert len({*EXIT.values()}) == len(EXIT), "三类错误必须能靠退出码区分"


def test_gate_lines_converts_gap_to_samples():
    """门禁失败要说清「差几条」——一个裸差值 0.0571 看不出是差一条还是差得远。"""
    from evals.run import gate_lines

    metrics = {"context_recall": 0.6429, "null_rate": 0.12, "n": 30, "n_context": 28}
    lines = gate_lines(metrics, {"context_recall": 0.70, "null_rate": 0.10})

    assert len(lines) == 2
    assert "还差约 2 个样本" in lines[0], "0.0571 × 28 = 1.6，进位应为 2 条"
    assert "样本" not in lines[1], "null_rate 的分母是判官调用数，换算成样本是错的"
    assert gate_lines({"context_recall": 0.70, "n_context": 28}, {"context_recall": 0.70}) == [], "刚好达标不算失败"
    assert "没有有效值" in gate_lines({"context_recall": None}, {"context_recall": 0.7})[0]


def test_latest_report_skips_ablation(tmp_path):
    """默认基线不能取消融报告：它是 `variants` 结构，没有 `metrics` 键。"""
    from evals.run import latest_report

    (tmp_path / "20260101-000000-ablation.json").write_text("{}", encoding="utf-8")
    (tmp_path / "20260101-000000-eval.json").write_text("{}", encoding="utf-8")
    assert latest_report(tmp_path).name.endswith("-eval.json")

    (tmp_path / "20260101-000000-eval.json").unlink()
    assert latest_report(tmp_path) is None, "只剩消融报告时应返回 None，而不是拿它当基线"
