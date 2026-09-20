#!/usr/bin/env python3
"""R2 reference backfill: generate reference answers for the narrative eval set.

The ``rag_evaluation_jobs`` queue carries a ``reference`` column and the scorer
(tools/evaluation_pipeline.py) routes reference-bearing jobs to the
reference-based metric panel. The historical 100-question narrative eval set
(tools/data/apple_narrative_questions_100.json) has no reference answers yet —
this script backfills them:

1. Parse the question JSON (adaptive: ``question`` / ``question_text`` /
   ``query`` fields; dict-with-``questions`` or bare list payloads).
2. Per question (default ``--with-context``): retrieve top-k contexts via the
   existing hybrid retrieval service (tools/llamaindex_retrieval.py), then ask
   the LLM for an evidence-grounded reference answer. Without
   ``--no-with-context`` the LLM answers the bare question (mode=question_only).
3. Per-question atomic checkpoint (tools/data/reference_answers_progress.json);
   re-runs skip already-completed questions.
4. Output tools/data/narrative_reference_answers.json:
   ``[{question, reference, model, mode, generated_at}, ...]`` plus a
   success/failure summary.

Usage (from repo root):
  venv/Scripts/python src/agent/scripts/generate_reference_answers.py --limit 2
  venv/Scripts/python src/agent/scripts/generate_reference_answers.py \
      --questions src/agent/tools/data/apple_narrative_questions_100.json \
      --concurrency 3 --enqueue --document-ids 9801
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

REPO_ROOT = AGENT_ROOT.parent
DEFAULT_QUESTIONS = AGENT_ROOT / "tools" / "data" / "apple_narrative_questions_100.json"
DEFAULT_OUTPUT = AGENT_ROOT / "tools" / "data" / "narrative_reference_answers.json"
DEFAULT_PROGRESS = AGENT_ROOT / "tools" / "data" / "reference_answers_progress.json"
# Lightweight plan-endpoint chat model (same default as gen_multimodal_evalset.py).
DEFAULT_MODEL = "openai/glm-5.3-flash"

_RETRIEVE_TIMEOUT_S = 300.0

_PROMPT_WITH_CONTEXT = """你是财务报告问答评测的标注员。下面是针对给定问题检索到的财报片段。\
请基于且仅基于这些片段，撰写一条标准参考答案（reference），供后续自动评测对齐使用。

问题：{question}

检索片段：
{contexts}

硬性要求：
1. 事实性：只使用片段中出现的信息，禁止编造片段之外的数字、日期或结论。
2. 保留片段中的关键数字（金额、百分比、期间、日期），与原文一致。
3. 简洁：不超过 150 词（英文作答）或 150 字（中文作答）。
4. 直接输出参考答案正文：不要任何前后缀说明、不要 JSON、不要引用标记。"""

_PROMPT_QUESTION_ONLY = """你是财务报告问答评测的标注员。请为给定问题直接撰写一条标准参考答案\
（reference），供后续自动评测对齐使用。

问题：{question}

硬性要求：
1. 事实性：给出业内公认的事实与数字；不确定的内容宁可省略，禁止编造。
2. 保留关键数字（金额、百分比、期间、日期）。
3. 简洁：不超过 150 词（英文作答）或 150 字（中文作答）。
4. 直接输出参考答案正文：不要任何前后缀说明、不要 JSON。"""


# ---------------- pure helpers (unit-tested offline) ----------------


def parse_questions(payload: object) -> list[dict]:
    """Normalize a question file into ``[{id, question}, ...]``.

    Accepts ``{"questions": [...]}`` dicts or bare lists; per-item question text
    may live in ``question`` / ``question_text`` / ``query`` and the id in
    ``id`` / ``qid`` / ``question_id`` (fallback: 1-based index).
    """
    raw: list[object]
    if isinstance(payload, dict):
        raw = payload.get("questions") or []  # type: ignore[union-attr]
    elif isinstance(payload, list):
        raw = payload
    else:
        return []
    items: list[dict] = []
    for idx, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            continue
        text = ""
        for key in ("question", "question_text", "query"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        if not text:
            continue
        qid = ""
        for key in ("id", "qid", "question_id"):
            value = entry.get(key)
            if value is not None and str(value).strip():
                qid = str(value).strip()
                break
        items.append({"id": qid or str(idx), "question": text})
    return items


def extract_reference(text: str | None) -> str:
    """Clean a raw LLM completion down to the reference answer text.

    Strips ``<think>`` reasoning blocks and markdown fences; if the model
    answered with a JSON object, pulls the ``reference`` / ``answer`` /
    ``response`` field.
    """
    if not text:
        return ""
    t = re.sub(r"(?is)<think>.*?</think>", "", str(text)).strip()
    t = re.sub(r"(?s)^```[a-zA-Z0-9_-]*[ \t]*\n?|\n?[ \t]*```$", "", t).strip()
    if t.startswith("{"):
        try:
            data = json.loads(t)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("reference", "answer", "response"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return t


def build_reference_prompt(question: str, contexts: list[str] | None) -> str:
    """Render the generation prompt; ``contexts=None/[]`` → question-only mode."""
    if contexts:
        joined = "\n\n".join(f"[片段{i}] {c}" for i, c in enumerate(contexts, start=1))
        return _PROMPT_WITH_CONTEXT.format(question=question, contexts=joined)
    return _PROMPT_QUESTION_ONLY.format(question=question)


def load_progress(path: Path) -> dict[str, dict]:
    """Read the checkpoint file tolerantly (corrupt/missing → empty dict)."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def save_progress(path: Path, progress: dict[str, dict]) -> None:
    """Atomic full-file checkpoint write (tempfile + os.replace, m4 pattern)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def select_pending(items: list[dict], progress: dict[str, dict]) -> list[dict]:
    """Questions without a completed checkpoint entry (failed ones re-run)."""
    done = {qid for qid, rec in progress.items() if rec.get("status") == "ok"}
    return [it for it in items if it["id"] not in done]


def build_output_records(progress: dict[str, dict], items: list[dict]) -> list[dict]:
    """Completed records in eval-set order, in the documented output shape."""
    records: list[dict] = []
    for it in items:
        rec = progress.get(it["id"]) or {}
        if rec.get("status") != "ok" or not str(rec.get("reference") or "").strip():
            continue
        records.append(
            {
                "question": rec.get("question") or it["question"],
                "reference": rec["reference"],
                "model": rec.get("model"),
                "mode": rec.get("mode"),
                "generated_at": rec.get("generated_at"),
            }
        )
    return records


def parse_document_ids(raw: str | None) -> list[int]:
    """``"9801,9805-9807"`` → [9801, 9805, 9806, 9807]; empty → []."""
    if not raw:
        return []
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def document_ids_from_payload(payload: object) -> list[int]:
    """``document_id_range: [low, high]`` (m4 convention) → inclusive id list."""
    if isinstance(payload, dict):
        rng = payload.get("document_id_range")
        if isinstance(rng, list | tuple) and len(rng) == 2:
            try:
                low, high = int(rng[0]), int(rng[1])
            except (TypeError, ValueError):
                return []
            return list(range(low, high + 1)) if low <= high else []
    return []


# ---------------- IO adapters (monkeypatched in tests) ----------------


async def _retrieve_context(question: str, document_ids: list[int], top_k: int) -> list[str]:
    """Top-k node texts from the existing hybrid retrieval service."""
    from tools.llamaindex_retrieval import retrieval_service

    result = await retrieval_service.retrieve(query=question, document_ids=document_ids)
    nodes = result.get("nodes") or []
    texts: list[str] = []
    for node in nodes[:top_k]:
        text = str(node.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts


async def _generate_reference(question: str, contexts: list[str] | None, model: str) -> str:
    """LLM reference generation via config-driven get_llm (no hardcoded keys)."""
    from tools.llm import get_llm

    llm = get_llm(model, temperature=0.0)
    response = await llm.ainvoke(build_reference_prompt(question, contexts))
    content = getattr(response, "content", response)
    return extract_reference(content if isinstance(content, str) else str(content))


# ---------------- orchestration ----------------


def _fail_record(item: dict, model: str, mode: str, error: str) -> dict:
    return {
        "id": item["id"],
        "question": item["question"],
        "reference": "",
        "model": model,
        "mode": mode,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "failed",
        "error": error[:300],
    }


async def run_backfill(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    items = parse_questions(payload)
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    if not items:
        print(f"no questions parsed from {args.questions}", file=sys.stderr)
        return 2

    document_ids = parse_document_ids(args.document_ids) or document_ids_from_payload(payload)
    mode = "with_context" if args.with_context else "question_only"
    if args.with_context and not document_ids:
        print(
            "no document ids: the questions file lacks document_id_range and --document-ids was not given",
            file=sys.stderr,
        )
        return 2

    progress_path = Path(args.progress)
    progress = load_progress(progress_path)
    pending = select_pending(items, progress)
    print(
        f"reference backfill: {len(items)} question(s), mode={mode}, model={args.model}, "
        f"resume: {len(items) - len(pending)} done, {len(pending)} pending"
    )

    # Preflight: fail fast with a clear message when retrieval infra is down
    # instead of hanging on per-question retries.
    if args.with_context and pending:
        try:
            await asyncio.wait_for(
                _retrieve_context(pending[0]["question"], document_ids, args.top_k),
                timeout=_RETRIEVE_TIMEOUT_S,
            )
        except Exception as exc:
            print(
                f"[fatal] retrieval unreachable for preflight question {pending[0]['id']!r}: "
                f"{type(exc).__name__}: {exc}\n"
                "check Milvus (MILVUS_URI) / Postgres (DATABASE_URL) are running, then re-run "
                "(completed questions resume from checkpoint)",
                file=sys.stderr,
            )
            return 2

    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    fail_count = 0

    async def worker(item: dict) -> None:
        nonlocal fail_count
        async with semaphore:
            record: dict
            try:
                contexts: list[str] | None = None
                if args.with_context:
                    contexts = await asyncio.wait_for(
                        _retrieve_context(item["question"], document_ids, args.top_k),
                        timeout=_RETRIEVE_TIMEOUT_S,
                    )
                reference = await _generate_reference(item["question"], contexts, args.model)
                if not reference.strip():
                    raise ValueError("LLM returned an empty reference")
                record = {
                    "id": item["id"],
                    "question": item["question"],
                    "reference": reference,
                    "model": args.model,
                    "mode": mode,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "ok",
                    **({"contexts": contexts} if contexts else {}),
                }
            except Exception as exc:
                record = _fail_record(item, args.model, mode, f"{type(exc).__name__}: {exc}")
            async with lock:
                progress[item["id"]] = record
                save_progress(progress_path, progress)
        status = record.get("status")
        if status == "ok":
            print(f"  [ok] id={item['id']} {item['question'][:60]}...")
        else:
            fail_count += 1
            print(f"  [FAIL] id={item['id']}: {record.get('error')}", file=sys.stderr)

    await asyncio.gather(*(worker(item) for item in pending))

    records = build_output_records(progress, items)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_records = [it for it in items if (progress.get(it["id"]) or {}).get("status") == "ok"]
    failed = [it for it in items if (progress.get(it["id"]) or {}).get("status") != "ok"]
    print(
        f"done: success={len(ok_records)} failed={len(failed)} output={output_path} "
        f"checkpoint={progress_path}"
    )
    if failed:
        print("failed ids: " + ", ".join(it["id"] for it in failed))

    if args.enqueue:
        enqueued = await enqueue_records(progress, ok_records, document_ids)
        print(f"enqueued {enqueued} evaluation job(s) with reference")

    return 1 if failed else 0


async def enqueue_records(progress: dict[str, dict], items: list[dict], document_ids: list[int]) -> int:
    """Push completed questions into rag_evaluation_jobs with their reference."""
    from tools.node_repository import enqueue_evaluation_job

    enqueued = 0
    for it in items:
        rec = progress.get(it["id"]) or {}
        if rec.get("status") != "ok":
            continue
        await enqueue_evaluation_job(
            trace_id=f"ref_backfill_{uuid.uuid4().hex[:12]}",
            document_ids=document_ids,
            query=rec.get("question") or it["question"],
            answer=rec["reference"],
            context_json=[
                {"text": text} for text in (rec.get("contexts") or [])
            ],
            reference=rec["reference"],
            metadata={"source": "reference_backfill", "mode": rec.get("mode"), "model": rec.get("model")},
        )
        enqueued += 1
    return enqueued


def main() -> int:
    parser = argparse.ArgumentParser(description="R2: backfill reference answers for the narrative eval set")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="Question JSON path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output reference JSON path")
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS, help="Checkpoint JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Generation model (config-driven provider)")
    parser.add_argument(
        "--with-context",
        dest="with_context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retrieve contexts first (default on); --no-with-context answers the bare question",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Retrieved contexts per question (default 6)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N questions (smoke)")
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel LLM/retrieval workers (default 3)")
    parser.add_argument("--document-ids", type=str, default=None, help="Comma/range list; overrides document_id_range")
    parser.add_argument(
        "--enqueue",
        action="store_true",
        help="After generation, enqueue rag_evaluation_jobs carrying the reference",
    )
    args = parser.parse_args()
    return asyncio.run(run_backfill(args))


if __name__ == "__main__":
    sys.exit(main())
