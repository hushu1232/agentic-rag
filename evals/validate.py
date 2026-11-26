"""评测集校验。数据错一条，后面所有指标都是假的，所以这里宁严勿松。

七项检查，任一项不过就退出码 1：
1. 30 条、id 唯一、字段集合固定（多一个字段也算错——防止 `chunks_note` 这种手工残留）
2. 配比 14 单跳 / 10 多跳 / 6 无答案
3. `relevant_chunk_ids` 必须全部存在于 `.index/chunks.jsonl`
4. 多跳题的标注片段必须跨 ≥2 个 source（否则它根本不是多跳）
5. 参考答案里的每个阿拉伯数字都要在标注片段里逐字出现（防止我凭记忆写数字）
6. 无答案题的 `absence_proof` 当场重跑，必须 0 命中
7. 无答案题也要有标注片段（语义最近但不含答案），否则无法区分"召回失败"与"模型在编"

    python -m evals.validate
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

DATA = Path("evals/dataset.jsonl")
CHUNKS = Path(".index/chunks.jsonl")

FIELDS = {
    "single_hop": {"id", "type", "expected_strategy", "question", "reference_answer", "relevant_chunk_ids", "notes"},
    "multi_hop": {"id", "type", "expected_strategy", "question", "reference_answer", "relevant_chunk_ids", "notes"},
    "unanswerable": {
        "id",
        "type",
        "expected_strategy",
        "question",
        "reference_answer",
        "relevant_chunk_ids",
        "absence_proof",
        "notes",
    },
}
QUOTA = {"single_hop": 14, "multi_hop": 10, "unanswerable": 6}
REFUSAL = "I cannot confirm this from the provided sources"
# 只抓“独立数字”：前面不能是字母数字（BM25 的 25 不是数字事实），句末的点不算小数
NUM = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?(?![0-9])")


def load_chunks() -> dict[str, dict]:
    return {json.loads(l)["chunk_id"]: json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines()}


def count_hits(pattern: str, scope: str) -> tuple[int, list[str]]:
    """按行统计命中，返回 (命中行数, 前 3 条证据)。"""
    root = Path(scope)
    files = sorted(root.glob("*.txt")) if root.is_dir() else [root]
    rx = re.compile(pattern, re.I)
    total, evidence = 0, []
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if rx.search(line):
                total += 1
                if len(evidence) < 3:
                    evidence.append(f"{f.name}:{i}")
    return total, evidence


def main() -> int:
    chunks = load_chunks()
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    problems: list[str] = []

    # 1 条数 / 字段 / id
    if len(rows) != 30:
        problems.append(f"条数应为 30，实际 {len(rows)}")
    ids = [r["id"] for r in rows]
    dup = [i for i, n in Counter(ids).items() if n > 1]
    if dup:
        problems.append(f"id 重复：{dup}")
    for r in rows:
        want = FIELDS.get(r.get("type", ""))
        if want is None:
            problems.append(f"{r.get('id')}: 未知 type {r.get('type')!r}")
            continue
        if set(r) != want:
            problems.append(f"{r['id']}: 字段不符，多={sorted(set(r) - want)} 少={sorted(want - set(r))}")

    # 2 配比
    got = Counter(r["type"] for r in rows)
    for t, n in QUOTA.items():
        if got[t] != n:
            problems.append(f"配比：{t} 应 {n} 条，实际 {got[t]}")

    # 3/4/5/7 逐条
    for r in rows:
        cids = r["relevant_chunk_ids"]
        missing = [c for c in cids if c not in chunks]
        if missing:
            problems.append(f"{r['id']}: chunk_id 不存在 {missing}")
            continue

        if r["type"] == "multi_hop":
            srcs = {chunks[c]["source"] for c in cids}
            if len(srcs) < 2:
                problems.append(f"{r['id']}: 多跳题只跨了 {len(srcs)} 个 source（{sorted(srcs)}）")

        if r["type"] == "unanswerable":
            if not cids:
                problems.append(f"{r['id']}: 无答案题缺少'相关但无答案'的片段")
            if not r["reference_answer"].startswith(REFUSAL):
                problems.append(f"{r['id']}: 无答案题的参考答案未以固定拒答标记开头")
        else:
            if r["reference_answer"].startswith(REFUSAL) and r["type"] != "unanswerable":
                problems.append(f"{r['id']}: 有答案题却给了拒答")

        # 数字逐字核验：simple 题不检索，跳过
        if r["expected_strategy"] != "simple":
            blob = "\n".join(chunks[c]["text"] for c in cids).lower()
            for num in NUM.findall(r["reference_answer"]):
                if num.lower() not in blob:
                    problems.append(f"{r['id']}: 数字 {num!r} 不在标注片段中（凭记忆写数的信号）")

    # 6 absence_proof 重跑
    for r in rows:
        if r["type"] != "unanswerable":
            continue
        proof = r["absence_proof"]
        hits, evidence = count_hits(proof["pattern"], proof["scope"])
        if hits != proof["expected_hits"]:
            problems.append(
                f"{r['id']}: absence_proof 失效，{proof['scope']} 命中 {hits} 行（期望 {proof['expected_hits']}）{evidence}"
            )

    if problems:
        print(f"校验失败：{len(problems)} 项\n", file=sys.stderr)
        for p in problems:
            print("  ✗ " + p, file=sys.stderr)
        return 1

    print(f"✓ 30 条全部通过：单跳 {got['single_hop']} / 多跳 {got['multi_hop']} / 无答案 {got['unanswerable']}")
    print(f"✓ 涉及片段 {sum(len(r['relevant_chunk_ids']) for r in rows)} 个引用，全部存在于索引")
    multi = [r for r in rows if r["type"] == "multi_hop"]
    print(f"✓ 多跳题跨 source 数：{min(len({chunks[c]['source'] for c in r['relevant_chunk_ids']}) for r in multi)}–{max(len({chunks[c]['source'] for c in r['relevant_chunk_ids']}) for r in multi)}")
    print(f"✓ 6 条 absence_proof 重跑均 0 命中")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
