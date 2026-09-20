#!/usr/bin/env python3
"""T2.5 multimodal gold-set generator — LLM drafting + anti-gaming self-checks.

Pipeline (docs/MULTIMODAL_RAG_PART2_DESIGN.md §11.1 T2.5 + migration doc v1.34-37):

1. Stratified sampling over ``rag_multimodal`` points (book x kind, text/image).
2. LLM drafts question+reference per template — hop count is DESIGNED at
   generation time (the fed chunks define the gold, not reverse-engineered):
   - single_hop (60%): 1 chunk in, HitRate@k target;
   - aggregation (30%): 2-4 same-chapter chunks in, listing/synthesis questions,
     gold = the full contributing set (GoldRecall@k + MRR measure ordering);
   - cross_book (10%): two books' same-topic chunks, exercises filter semantics.
3. Anti-gaming prompt constraints: self-contained (no pronouns referring to
   "the passage"), never restate the answer's key wording, aggregation must
   need every fed chunk.
4. Non-LLM hard assertions baked into the gold fields: ``gold_chunk_ids`` +
   ``scope`` — retrieval correctness is judged WITHOUT an LLM (the M4
   false-positive lesson: judges only verify "context supports answer", not
   scope).
5. n-gram leak self-check: character 8-grams of the question appearing in the
   fed chunk text flag ``leak_suspect`` (human review list, not auto-drop).

Output: human-editable JSON draft at ``tools/data/multimodal_evalset_draft.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

DEFAULT_OUTPUT = Path("tools/data/multimodal_evalset_draft.json")
DEFAULT_MODEL = "openai/glm-5.3-flash"
_NGRAM = 8  # character n-gram for the leak self-check (Chinese-friendly)


def load_chunks(limit_per_book: int = 60, collection: str | None = None) -> list[dict]:
    """Read text/image chunks from a multimodal collection via Milvus query.

    ``collection`` defaults to ``config.multimodal_collection`` (CLI behavior);
    the testset-generation API passes the caller's dynamic collection through.
    """
    from core.config import config
    from tools.milvus_store import get_client

    target = (collection or config.multimodal_collection or "").strip()
    client = get_client()
    if not client.has_collection(target):
        raise SystemExit(f"collection {target!r} missing; ingest first")
    rows = client.query(
        collection_name=target,
        filter="document_id > 0",
        output_fields=[
            "document_id", "filename", "title", "kind", "page_no",
            "text", "category", "image_ref", "book_id", "chapter_label",
        ],
        limit=16384,
    )
    by_book: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_book[str(row.get("book_id") or row["document_id"])].append(
            {
                "chunk_id": str(row["id"]),
                "document_id": int(row["document_id"]),
                "filename": row.get("filename"),
                "title": row.get("title") or "",
                "kind": row.get("kind") or "text",
                "page_no": int(row.get("page_no") or 0),
                "text": str(row.get("text") or ""),
                "book_id": str(row.get("book_id") or row["document_id"]),
                "chapter_label": row.get("chapter_label") or "",
            }
        )
    sampled: list[dict] = []
    for book_id in sorted(by_book):
        sampled.extend(by_book[book_id][:limit_per_book])
    return sampled


def stratified_sample(chunks: list[dict], total: int, ratio: tuple[float, float, float]) -> dict[str, list[dict]]:
    """Split chunks into pools per question type, stratified by book x kind."""
    rng = random.Random(20260918)
    n_single = max(1, round(total * ratio[0]))
    n_aggr = max(0, round(total * ratio[1]))
    n_cross = max(0, round(total * ratio[2]))

    by_book_kind: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_book_kind[(chunk["book_id"], chunk["kind"])].append(chunk)
    pools = {k: rng.sample(v, len(v)) for k, v in by_book_kind.items()}

    def draw(n: int, group_size: int) -> list[list[dict]]:
        """Round-robin over book x kind cells, drawing ``group_size`` per group."""
        groups: list[list[dict]] = []
        cell_cycle = sorted(pools)
        made = True
        while made and len(groups) < n:
            made = False
            for cell in cell_cycle:
                if len(groups) >= n:
                    break
                pool = pools[cell]
                if len(pool) >= group_size:
                    groups.append(pool[:group_size])
                    del pool[:group_size]
                    made = True
        return groups

    return {
        "single_hop": draw(n_single, 1),
        "aggregation": draw(n_aggr, 3),
        "cross_book": _cross_book_pairs(pools, n_cross, rng),
    }


def _cross_book_pairs(pools: dict[tuple[str, str], list[dict]], n: int, rng: random.Random) -> list[list[dict]]:
    """Pair chunks from two DIFFERENT books (same kind), for comparison questions."""
    books = sorted({book for book, _kind in pools})
    pairs: list[list[dict]] = []
    if len(books) < 2:
        return pairs
    # Deterministic rotation: book[i] x book[i+1]
    consumable = {book: [c for (b, _k), cs in pools.items() if b == book for c in cs] for book in books}
    for i in range(len(books) - 1):
        if len(pairs) >= n:
            break
        left, right = consumable[books[i]], consumable[books[i + 1]]
        if left and right:
            pairs.append([left.pop(0), right.pop(0)])
    return pairs


_SINGLE_HOP_PROMPT = """你根据给定的文档片段出一条单跳事实问答题。

片段（chunk_id={chunk_id}，来自《{book}》）：
{context}

出题硬约束（违反即废题）：
1. 问题必须自包含：不得出现"该片段/上文/这段文字/图中"等指代词，读者只看问题就能理解问什么。
2. 问题中不得复述答案的关键词句（答案里的关键数字/术语原句不要照抄进问题）。
3. 答案必须能在片段中直接找到，reference 用一两句完整中文陈述并保留关键数字。
4. 只输出 JSON：{{"question": "...", "reference": "..."}}"""

_AGGREGATION_PROMPT = """你根据给定的多个同章文档片段出一条列举/综合题（多 gold）。

片段列表：
{context}

出题硬约束（违反即废题）：
1. 问题必须自包含，无任何指代词。
2. 问题不得复述答案中的关键词句。
3. 综合题必须需要全部片段才能完整回答（每个片段至少贡献一个答案要点），单片段可答的题不合格。
4. reference 按要点分条列出，保留关键数字。
5. 只输出 JSON：{{"question": "...", "reference": "要点1；要点2；要点3"}}"""

_CROSS_BOOK_PROMPT = """你根据两本书各一个片段出一条跨书比较/关联题。

片段A（来自《{book_a}》）：{context_a}
片段B（来自《{book_b}》）：{context_b}

出题硬约束（违反即废题）：
1. 问题自包含、无指代词；问题中点明两本书的主题来源（如"在《X》与《Y》中"）但不复述答案关键句。
2. 比较点必须在两个片段中都有对应内容支撑。
3. reference 分两书作答，保留关键数字。
4. 只输出 JSON：{{"question": "...", "reference": "..."}}"""


def _context_of(chunk: dict, max_chars: int = 1200) -> str:
    title = f"标题：{chunk['title']}\n" if chunk["title"] else ""
    return f"{title}{chunk['text'][:max_chars]}"


def _parse_json_response(text: str) -> dict | None:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _shared_runs(question: str, text: str, n: int = _NGRAM) -> list[str]:
    """Maximal contiguous shared spans (>= n chars, whitespace-insensitive).

    Folding the shared 8-gram set into maximal runs distinguishes ONE proper
    noun (e.g. "Data Artisans" → a single run) from multiple copied clauses.
    """
    q_clean = re.sub(r"\s+", "", question or "")
    t_clean = re.sub(r"\s+", "", text or "")
    q_grams = {q_clean[i : i + n] for i in range(len(q_clean) - n + 1)}
    runs: list[str] = []
    i = 0
    while i <= len(t_clean) - n:
        if t_clean[i : i + n] in q_grams:
            end = i + n  # run is t_clean[i:end]
            while end <= len(t_clean) - 1 and t_clean[end - n + 1 : end + 1] in q_grams:
                end += 1
            runs.append(t_clean[i:end])
            i = end
        else:
            i += 1
    return runs


def leak_check(question: str, fed_chunks: list[dict]) -> list[str]:
    """Flag chunks sharing >=2 DISTINCT verbatim runs (or one >=16-char run).

    Single proper-noun overlap is expected in self-contained questions and is
    NOT a leak (calibration finding, 2026-09-19: 6/12 false positives were
    names like Infoprompt/Structured Streaming); repeated occurrences of the
    same term count once.
    """
    leaking: list[str] = []
    for chunk in fed_chunks:
        runs = set(_shared_runs(question, chunk["text"]))
        long_runs = [r for r in runs if len(r) >= 16]
        if long_runs or len(runs) >= 2:
            leaking.append(chunk["chunk_id"])
    return leaking


async def draft_questions(groups: dict[str, list[list[dict]]], model: str | None = None) -> list[dict]:
    from tools.llm import get_llm

    llm = get_llm(model, temperature=0.3)
    questions: list[dict] = []

    async def call(prompt: str) -> dict | None:
        try:
            response = await llm.ainvoke(prompt)
            return _parse_json_response(response.content)
        except Exception as exc:
            print(f"  [warn] LLM call failed: {exc}", file=sys.stderr)
            return None

    for group in groups.get("single_hop", []):
        chunk = group[0]
        prompt = _SINGLE_HOP_PROMPT.format(
            chunk_id=chunk["chunk_id"], book=chunk["book_id"], context=_context_of(chunk)
        )
        data = await call(prompt)
        if not data or not data.get("question") or not data.get("reference"):
            continue
        questions.append(
            {
                "type": "single_hop",
                "question": data["question"],
                "reference": data["reference"],
                "gold_chunk_ids": [chunk["chunk_id"]],
                "scope": {"book_id": chunk["book_id"], "chapter_document_id": chunk["document_id"]},
                "gold_kinds": [chunk["kind"]],
                "provenance": [
                    {"chunk_id": chunk["chunk_id"], "book_id": chunk["book_id"], "page_no": chunk["page_no"]}
                ],
            }
        )

    for group in groups.get("aggregation", []):
        context = "\n\n".join(
            f"- chunk_id={c['chunk_id']}（《{c['book_id']}》）：{_context_of(c, 600)}" for c in group
        )
        data = await call(_AGGREGATION_PROMPT.format(context=context))
        if not data or not data.get("question") or not data.get("reference"):
            continue
        questions.append(
            {
                "type": "aggregation",
                "question": data["question"],
                "reference": data["reference"],
                "gold_chunk_ids": [c["chunk_id"] for c in group],
                "scope": {"book_id": group[0]["book_id"], "chapter_document_id": group[0]["document_id"]},
                "gold_kinds": sorted({c["kind"] for c in group}),
                "provenance": [
                    {"chunk_id": c["chunk_id"], "book_id": c["book_id"], "page_no": c["page_no"]} for c in group
                ],
            }
        )

    for group in groups.get("cross_book", []):
        if len(group) < 2 or group[0]["book_id"] == group[1]["book_id"]:
            continue
        data = await call(
            _CROSS_BOOK_PROMPT.format(
                book_a=group[0]["book_id"],
                book_b=group[1]["book_id"],
                context_a=_context_of(group[0], 600),
                context_b=_context_of(group[1], 600),
            )
        )
        if not data or not data.get("question") or not data.get("reference"):
            continue
        questions.append(
            {
                "type": "cross_book",
                "question": data["question"],
                "reference": data["reference"],
                "gold_chunk_ids": [c["chunk_id"] for c in group],
                "scope": {"book_id": None, "chapter_document_id": None},  # two books: scope = both
                "scope_books": [group[0]["book_id"], group[1]["book_id"]],
                "gold_kinds": ["text"],
                "provenance": [
                    {"chunk_id": c["chunk_id"], "book_id": c["book_id"], "page_no": c["page_no"]} for c in group
                ],
            }
        )

    return questions


async def generate_evalset_core(
    *,
    collection: str | None = None,
    total: int = 18,
    ratio: tuple[float, float, float] = (0.6, 0.3, 0.1),
    max_chunks_per_book: int = 60,
    model: str = DEFAULT_MODEL,
    progress=None,
) -> dict:
    """Importable T2.5 core: sample → draft → leak-check → payload dict (§11.1).

    ``collection`` routes the sampling query (None = config default);
    ``progress`` is an optional callable receiving the CLI progress lines.
    Raises ValueError for client-rejectable problems (no chunks / unknown
    collection — the latter surfaces as SystemExit inside load_chunks and is
    normalized here). The returned payload matches the CLI draft JSON exactly.
    """
    say = progress or (lambda _msg: None)

    say("[1/3] loading chunks from Milvus ...")
    try:
        # Sync Milvus pull of up to 16384 rows → worker thread: this core runs
        # as a background API job, and the call must not block the event loop
        # (review P1-1).
        chunks = await asyncio.to_thread(
            load_chunks, limit_per_book=max_chunks_per_book, collection=collection
        )
    except SystemExit as exc:  # load_chunks signals unknown collection via SystemExit
        raise ValueError(str(exc)) from None
    if not chunks:
        raise ValueError("no chunks found; run ingest_multimodal_pdf.py first")
    books = sorted({c["book_id"] for c in chunks})
    say(f"      {len(chunks)} chunks across {len(books)} books: {books}")
    if len(books) < 2:
        say("      [warn] only one book ingested; cross_book questions will be skipped")

    say(f"[2/3] drafting questions via LLM ({model}) ...")
    groups = stratified_sample(chunks, total, tuple(ratio))
    questions = await draft_questions(groups, model=model)

    # Leak self-check needs full chunk text; re-attach from the source pools.
    by_id = {c["chunk_id"]: c for c in chunks}
    leak_count = 0
    for item in questions:
        fed = [by_id[cid] for cid in item["gold_chunk_ids"] if cid in by_id]
        leaking = leak_check(item["question"], fed)
        item["leak_suspect"] = bool(leaking)
        item["leak_chunk_ids"] = leaking
        leak_count += bool(leaking)

    say("[3/3] writing draft ...")
    return {
        "version": "t2.5-draft-1",
        "generator": {"model": model, "seed": 20260918, "ratio": list(tuple(ratio))},
        "counts": {
            "total": len(questions),
            "single_hop": sum(1 for q in questions if q["type"] == "single_hop"),
            "aggregation": sum(1 for q in questions if q["type"] == "aggregation"),
            "cross_book": sum(1 for q in questions if q["type"] == "cross_book"),
            "leak_suspect": leak_count,
        },
        "questions": questions,
    }


async def main_async(args: argparse.Namespace) -> None:
    from tools.rag_stage_log import log_rag

    try:
        payload = await generate_evalset_core(
            total=args.total,
            ratio=tuple(args.ratio),
            max_chunks_per_book=args.max_chunks_per_book,
            model=args.model,
            progress=print,
        )
    except ValueError as exc:  # CLI parity: same message, same exit code as before
        raise SystemExit(str(exc)) from None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    leak_count = payload["counts"]["leak_suspect"]
    log_rag(
        "mm_evalset_generated",
        total=payload["counts"]["total"],
        leak_suspect=leak_count,
        output=str(args.output),
    )
    print(json.dumps(payload["counts"], ensure_ascii=False))
    print(f"draft written to {args.output} — 人工校准后删除坏题/修正答案再交付评测")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multimodal gold-set draft (T2.5)")
    parser.add_argument("--total", type=int, default=18, help="Target question count (default 18)")
    parser.add_argument(
        "--ratio", type=float, nargs=3, default=(0.6, 0.3, 0.1), help="single_hop/aggregation/cross_book ratios"
    )
    parser.add_argument("--max-chunks-per-book", type=int, default=60)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Drafting model (default glm-5.3-flash: same plan endpoint, much faster than glm-5.3)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
