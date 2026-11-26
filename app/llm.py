"""模型出口：全项目唯一允许调用 LLM 的地方。

重试、超时、token 统计都收敛在这里，节点拿到的只有字符串或已校验的 pydantic 对象。
`LlmProto` 是测试注入点——5 个回归测试用假模型跑，不联网、不耗 token，秒级完成。

为什么不用 SDK 自带重试：`max_retries=0` 传进去由这里统一控制，是为了让「调用了几次」
这个事实可被断言（`test_loop_guard` 要数检索次数，同理要数模型次数）。
"""

from __future__ import annotations

import time
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Config

T = TypeVar("T", bound=BaseModel)

ATTEMPTS = 2      # 首次 + 1 次重试；结构化解析失败同样是"首次 + 1 次重试"
BACKOFF = 0.5     # 秒，指数退避基数
TIMEOUT = 60.0    # 秒


class LlmError(Exception):
    """模型层失败：网络、超时、或结构化输出解析不出来。CLI 映射为可读提示 + 退出码 1。"""


class LlmProto(Protocol):
    """测试用的假模型只需满足这三个成员。节点依赖它，不依赖具体实现。"""

    @property
    def tokens(self) -> int: ...

    def complete(self, prompt: str, system: str | None = None) -> str: ...

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T: ...


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


class Llm:
    """真实实现。`client` 参数存在只为一件事：测试里塞一个假客户端，不联网。"""

    def __init__(self, cfg: Config, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                timeout=TIMEOUT,
                max_retries=0,
            )
        self.cfg = cfg
        self._client = client
        self._tokens = 0
        self.calls = 0  # 供测试断言"调用了几次模型"

    @property
    def tokens(self) -> int:
        return self._tokens

    def _chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        last: Exception | None = None
        for attempt in range(ATTEMPTS):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.cfg.model,
                    "messages": messages,
                    "temperature": 0,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self._client.chat.completions.create(**kwargs)
                self.calls += 1
                self._tokens += int(getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0)
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # 网络/超时/鉴权，统一包装
                last = exc
                if attempt + 1 < ATTEMPTS:
                    time.sleep(BACKOFF * (2**attempt))
        raise LlmError(f"模型调用失败（model={self.cfg.model}）：{last}") from last

    def complete(self, prompt: str, system: str | None = None) -> str:
        return self._chat(_messages(prompt, system))

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        """结构化输出：解析失败重试 1 次，仍失败抛 LlmError（绝不返回半成品）。"""
        fields = ", ".join(schema.model_fields)
        ask = f"{prompt}\n\n只输出一个 JSON 对象，必须包含字段：{fields}。不要输出解释文字。"
        text = self._chat(_messages(ask, system), json_mode=True)
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            reason = exc.errors()[0]["msg"] if exc.errors() else str(exc)
            retry = f"{ask}\n\n上次输出无法解析：{reason}。请只输出合法 JSON，字段齐全。"
            text = self._chat(_messages(retry, system), json_mode=True)
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc2:
                raise LlmError(f"结构化输出解析失败（{schema.__name__}）：{exc2}") from exc2
