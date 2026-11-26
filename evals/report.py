"""报告渲染。每次评测同时产出 json（给 diff 用）与 md（给人看）。

json 必须自带 `meta`（embedder / 模型 / 数据集哈希 / 条数），否则过两天看到一份
`context_recall=0.68` 的报告，无法判断它是真实语义向量跑的还是 hash 占位跑的——
这两种数字放一起比较是错的，必须在报告头就拦住。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# (键, 中文名, 方向)。反向指标越大越差。
METRIC_NAMES = [
    ("context_recall", "context_recall", "up"),
    ("groundedness", "groundedness", "up"),
    ("answer_relevance", "answer_relevance", "up"),
    ("refusal_accuracy", "refusal_accuracy", "up"),
    ("routing_accuracy", "routing_accuracy", "up"),
    ("null_rate", "null_rate", "down"),
]


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def _delta(cur: float | None, base: float | None, direction: str) -> str:
    if cur is None or base is None:
        return "—"
    d = cur - base
    if abs(d) < 1e-9:
        return "0"
    good = d > 0 if direction == "up" else d < 0
    return f"{d:+.4f} {'↑' if good else '↓'}"


def diff_rows(base: dict, cur: dict) -> list[tuple[str, str, str, str]]:
    return [
        (name, _fmt(base.get("metrics", {}).get(key)), _fmt(cur["metrics"].get(key)), _delta(
            cur["metrics"].get(key), base.get("metrics", {}).get(key), direction
        ))
        for key, name, direction in METRIC_NAMES
    ]


def ablation_rows(results: list[dict]) -> list[tuple[str, str, str, str, str]]:
    """四组消融：每行一个变体，列是指标；Δ 相对第一组（单路向量）算。"""
    if not results:
        return []
    first = results[0]["metrics"]
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append(
            (
                r["label"],
                _fmt(m.get("context_recall")),
                _delta(m.get("context_recall"), first.get("context_recall"), "up"),
                _fmt(m.get("groundedness")),
                _delta(m.get("groundedness"), first.get("groundedness"), "up"),
            )
        )
    return rows


def save_ablation(results: list[dict], meta: dict, gates: dict[str, float], label: str = "ablation") -> tuple[Path, Path]:
    """四组消融落盘。**不能只打印到终端**——README 与 resume 都指向 `reports/ablation-*.md`，
    只打印等于承诺了不存在的文件；而且趋势要能被事后翻出来对比，终端输出会随关闭窗口消失。
    """
    out_dir = Path(meta["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"{stamp}-{label}"

    payload = {"label": label, "meta": meta, "gates": gates, "variants": results}
    base.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# 消融报告 · {label}",
        "",
        f"- 时间：{meta['ran_at']}",
        f"- 样本：{(results[0]['metrics']['n'] if results else 0)} 条",
        f"- embedder：`{meta['embedder']}` ｜ 生成模型：`{meta['model']}` ｜ 判官：`{meta['judge_model']}`",
        f"- 数据集：`{meta['dataset']}`（sha1 {meta['dataset_sha1'][:12]}）",
        "",
        "每组只比上一组多一个开关：A 只看向量 → B 加 BM25 但**只是拼接**（源实现形态）→ "
        "C 换成 RRF → D 打开多跳。这样每一行的 Δ 都只归属一个变量。",
    ]
    if meta["embedder"] == "hash":
        lines += [
            "",
            "> **警告**：`hash` 占位向量下，Δ 的方向可以参考，**绝对值不可引用**。",
            "> 切到 `openai` 后必须重跑，不能拿本表与语义向量的表拼接。",
        ]
    lines += ["", "| 变体 | context_recall | Δ | groundedness | Δ |", "|---|---:|---:|---:|---:|"]
    for row in ablation_rows(results):
        lines.append("| " + " | ".join(row) + " |")

    md = base.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base.with_suffix(".json"), md


def save(metrics: dict, rows: list[dict], meta: dict, gates: dict[str, float], label: str) -> tuple[Path, Path]:
    out_dir = Path(meta["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"{stamp}-{label}"

    payload = {"label": label, "meta": meta, "metrics": metrics, "gates": gates, "rows": rows}
    base.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# 评测报告 · {label}",
        "",
        f"- 时间：{meta['ran_at']}",
        f"- 样本：{metrics['n']} 条（其中参与 context_recall 的 {metrics['n_context']} 条）",
        f"- embedder：`{meta['embedder']}` ｜ 生成模型：`{meta['model']}` ｜ 判官：`{meta['judge_model']}`",
        f"- 数据集：`{meta['dataset']}`（sha1 {meta['dataset_sha1'][:12]}）",
    ]
    if meta["embedder"] == "hash":
        lines += [
            "",
            "> **警告**：本次用 `hash` 占位向量，只有词面相似度，没有语义泛化。",
            "> 这份报告可以证明链路通、可以对比同一 embedder 下的消融，但**绝对值不能写进简历**。",
        ]

    lines += ["", "## 指标", "", "| 指标 | 值 | 门禁 | 结论 |", "|---|---:|---:|---|"]
    for key, name, direction in METRIC_NAMES:
        v, thr = metrics.get(key), gates.get(key)
        if v is None or thr is None:
            verdict = "—"
        else:
            ok = v <= thr if direction == "down" else v >= thr
            verdict = "PASS" if ok else f"FAIL（差 {abs(v - thr):.4f}）"
        lines.append(f"| {name} | {_fmt(v)} | {_fmt(thr)} | {verdict} |")

    failed = sorted(
        [r for r in rows if r["flags"]],
        key=lambda r: (-len(r["flags"]), r.get("recall") if r.get("recall") is not None else 1.0),
    )
    # 只列前 5：30 条里失败 15 条时全列会把 md 变成日志，反而没人看。json 里是全量。
    shown = failed[:5]
    lines += ["", f"## 失败样本（{len(failed)} / {len(rows)} 条）", ""]
    if shown:
        if len(failed) > len(shown):
            lines.append(f"按失分排序只列前 {len(shown)} 条，全量见同名 json。")
            lines.append("")
        lines += ["| id | 类型 | 问题 | 说明 |", "|---|---|---|---|"]
        for r in shown:
            q = r["question"].replace("|", "\\|")[:70]
            lines.append(f"| {r['id']} | {r['type']} | {q} | {'; '.join(r['flags'])} |")
    else:
        lines.append("无。")

    # 节点耗时：从 trace 汇总。没有这张表，"慢在哪一步"就只能靠猜——而它恰恰是
    # 多跳回边最容易被质疑的地方（"你多跳一轮多花多少毫秒"）。
    node_ms: dict[str, list[int]] = {}
    for r in rows:
        for step in r.get("trace", []):
            node_ms.setdefault(step["node"], []).append(step["ms"])
    if node_ms:
        lines += ["", "## 节点耗时", "", "| 节点 | 调用次数 | 平均 ms | 最大 ms |", "|---|---:|---:|---:|"]
        for node, ms in sorted(node_ms.items(), key=lambda kv: -sum(kv[1])):
            lines.append(f"| {node} | {len(ms)} | {sum(ms) / len(ms):.1f} | {max(ms)} |")
    lines += ["", f"- token 总量：{metrics.get('tokens', 0):,}"]

    md = base.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base.with_suffix(".json"), md
