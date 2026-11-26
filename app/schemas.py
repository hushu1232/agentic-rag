"""跨节点契约。所有结构在此声明，节点只允许读写这里定义过的字段。

`extra="forbid"` 是刻意的：源实现里 `retrieveDocs` 写进了 `retrievedDocs` 的状态，
LangGraph 静默丢弃未知键，下游拿到空数组还以为是"没召回"，排查了两轮。这里直接报错。

分三层候选是刻意的（ADR-011）：`candidates` 空 = 召回失败，`fused` 缺正确片段 = 融合
排错，`selected` 缺 = 截断截掉。合成一层就无法区分这三种失效。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Strategy(str, Enum):
    simple = "simple"
    complex = "complex"


class NextAction(str, Enum):
    retrieve = "retrieve"
    generate = "generate"


class Out(BaseModel):
    """模型结构化输出的基类：多一个字段就报错，避免"看着解析成功其实字段丢了"。"""

    model_config = ConfigDict(extra="forbid")


class RouteOut(Out):
    strategy: Strategy
    reason: str


class SubQuestionsOut(Out):
    """上限 8 与源实现一致；超出的部分由节点保序截断到 `max_sub_questions`。"""

    sub_questions: list[str] = Field(min_length=1, max_length=8)
    reason: str = ""


class PlanOut(Out):
    next_action: NextAction
    reason: str = ""


class Chunk(BaseModel):
    """索引里的最小单位。`chunk_id` 由 `source#序号` 生成，是评测标注的唯一锚点。"""

    chunk_id: str
    source: str
    heading: str
    text: str


class Candidate(Chunk):
    """召回结果。三路分数分开存，融合只认 `rrf`，不覆盖原始分（否则无法诊断）。"""

    vector_score: float | None = None
    bm25_score: float | None = None
    rrf: float = 0.0
    rerank_score: float | None = None  # 仅 rerank 开启时写入


class TraceStep(BaseModel):
    node: str
    ms: int
    decision: str = ""
    detail: str = ""


class State(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    strategy: Strategy | None = None
    route_reason: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    next_idx: int = 0
    current_query: str = ""
    rounds: int = 0
    max_rounds: int = 3
    next_action: NextAction | None = None  # plan 写，条件边读——唯一入口，杜绝读错字段
    candidates: list[Candidate] = Field(default_factory=list)
    fused: list[Candidate] = Field(default_factory=list)
    selected: list[Candidate] = Field(default_factory=list)
    context: str = ""
    answer: str = ""
    trace: list[TraceStep] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)  # decompose_degraded / retrieval_degraded
    tokens: int = 0


class RunResult(BaseModel):
    question: str
    answer: str
    strategy: Strategy | None = None
    selected: list[Candidate] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    tokens: int = 0
    elapsed_ms: int = 0
