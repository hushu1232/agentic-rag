"""重排：一个可关的旁路。

默认 `enabled=false` 直通 `fused[:top_n]`. 理由不是"重排没用"，而是**这一层必须先有
基线再打开**：关着跑一次，打开再跑一次，两次的 `context_recall` 差才是重排的贡献
（`--ablation` 里的一行）。源实现把 DashScope rerank 直接写进主链路，因此永远说不清
"这个 0.7 是检索好还是重排好"。

接口按 DashScope text-rerank 的形态写，但不绑定它：`poster` 可注入，测试不联网。
"""

from __future__ import annotations

import json
import urllib.request

from app.schemas import Candidate


class RerankError(RuntimeError):
    pass


def _post(url: str, payload: dict, api_key: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - 配置里的固定 URL
        return json.loads(resp.read())


class Reranker:
    def __init__(self, cfg, poster=_post) -> None:
        self.cfg = cfg
        self.poster = poster

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        """关闭时原样切片（逐位相同，可断言）；开启时按服务返回的 index 回取并重排。"""
        if not self.cfg.rerank.enabled:
            return candidates[:top_n]
        if not candidates:
            return []

        payload = {
            "model": self.cfg.rerank.model,
            "input": {"query": query, "documents": [c.text for c in candidates]},
            "parameters": {"top_n": top_n, "return_documents": False},
        }
        try:
            data = self.poster(self.cfg.rerank.url, payload, self.cfg.rerank_api_key)
        except Exception as exc:
            # 重排失败不该让整次回答失败：退到融合顺序，并让调用方记一条降级。
            raise RerankError(f"重排请求失败：{exc}") from exc

        results = (data.get("output") or {}).get("results") or []
        picked: list[Candidate] = []
        for item in results[:top_n]:
            idx = item.get("index")
            if not isinstance(idx, int) or not 0 <= idx < len(candidates):
                raise RerankError(f"重排返回的下标非法：{idx!r}（共 {len(candidates)} 条候选）")
            cand = candidates[idx].model_copy()
            cand.rerank_score = float(item.get("relevance_score", 0.0))
            picked.append(cand)
        if not picked:
            raise RerankError("重排返回空结果")
        return picked
