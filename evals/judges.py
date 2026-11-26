"""三个判官。全部走结构化输出，失败一律记 `None`，绝不返回 0 或 1。

为什么不允许"判官挂了就当 0 分"：那会让指标看起来是模型变差了，实际是 API 抖了。
`null_rate` 因此是独立指标并且是反向门禁（超过 10% 直接 fail），而不是把 null 悄悄剔掉
——剔掉等于让"判官挂了"变成"这条题不存在"，指标反而更好看。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.config import Config
from app.llm import Llm, LlmError
from app.schemas import Out

GROUNDED = """Judge whether the ANSWER is supported by the PASSAGES.

2 = every factual claim in the answer is stated in the passages
1 = the answer is mostly supported but contains at least one claim not in the passages
0 = the answer's main claim is absent from, or contradicted by, the passages

A refusal that names a missing fact counts as 2 when the passages indeed do not contain it.

Reply with JSON only: {"score": 0 | 1 | 2, "reason": "<one sentence>"}"""

RELEVANT = """Judge whether the ANSWER actually addresses the QUESTION.

2 = directly answers what was asked
1 = partially answers, or answers a nearby question
0 = off-topic

Reply with JSON only: {"score": 0 | 1 | 2, "reason": "<one sentence>"}"""

REFUSAL_CHECK = """The answer begins with the required refusal marker. Check whether it still sneaks in a guess.

refused = true  -> it only states that the corpus does not contain the fact, with no invented detail
refused = false -> after the marker it supplies a concrete number, name, or date that it cannot have from the passages

Reply with JSON only: {"refused": true | false, "reason": "<one sentence>"}"""


class ScoreOut(Out):
    score: int = Field(ge=0, le=2)
    reason: str = ""


class RefusalOut(Out):
    refused: bool
    reason: str = ""


class Judges:
    """`llm` 可注入假对象（测试）。未注入时用 `judge_model` 单独构造，与生成模型解耦。"""

    def __init__(self, cfg: Config, llm: Any | None = None) -> None:
        self.cfg = cfg
        self.llm = llm if llm is not None else Llm(cfg.model_copy(update={"model": cfg.judge_model}))
        self.total = 0
        self.null = 0

    def _ask(self, prompt: str, schema: type, payload: str) -> Any | None:
        self.total += 1
        try:
            return self.llm.complete_structured(prompt, schema, payload)
        except (LlmError, ValueError):
            self.null += 1
            return None

    def groundedness(self, question: str, answer: str, context: str) -> float | None:
        out = self._ask(GROUNDED, ScoreOut, f"PASSAGES:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer}")
        return None if out is None else out.score / 2

    def relevance(self, question: str, answer: str) -> float | None:
        out = self._ask(RELEVANT, ScoreOut, f"QUESTION: {question}\n\nANSWER: {answer}")
        return None if out is None else out.score / 2

    def refusal_reviewed(self, question: str, answer: str) -> bool | None:
        """只用于「已带拒答标记」的样本：复核它有没有在标记后面偷偷编。"""
        out = self._ask(REFUSAL_CHECK, RefusalOut, f"QUESTION: {question}\n\nANSWER: {answer}")
        return None if out is None else out.refused
