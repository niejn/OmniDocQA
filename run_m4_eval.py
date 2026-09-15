"""M4 evaluation: run 100 questions sequentially with checkpoint/resume.

- Checkpoint file (JSON) records per-question status; written atomically after
  EVERY question, so a crash/Ctrl-C loses at most one question.
- Re-running with the same --checkpoint skips questions already "ok";
  --retry-failed re-runs the failed ones.
- The scoring phase (scripts/run_evaluate_pending_parallel.py) resumes on its
  own from the rag_evaluation_jobs queue — no checkpoint needed there.

Usage:
  python run_m4_eval.py --checkpoint m4_progress_roundA_os.json
  python run_m4_eval.py --checkpoint m4_progress_roundA_os.json --limit 2   # trial run, resumable
  python run_m4_eval.py --checkpoint m4_progress_roundA_os.json --retry-failed
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import httpx

QUESTIONS_PATH = "src/agent/tools/data/apple_narrative_questions_100.json"
BASE_URL = "http://127.0.0.1:8000"
ASK_PATH = "/agent/api/ask/generate"
TIMEOUT = 180.0
ATTEMPTS_PER_QUESTION = 3  # first try + 2 retries (LLM rate limits / reranker warmup)
RETRY_DELAY_S = 5.0


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            print(f"[warn] checkpoint {path} unreadable ({exc}); starting fresh")
    return {}


def save_checkpoint(path: Path, cp: dict) -> None:
    """Atomic write: temp file in the same dir + os.replace (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cp, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


async def ask_once(client: httpx.AsyncClient, base_url: str, question: str, document_ids: list) -> tuple[bool, str]:
    try:
        resp = await client.post(
            f"{base_url}{ASK_PATH}",
            json={
                "question": question,
                "document_ids": document_ids,
                "top_k": 3,
                "detail_level": "detailed",
                "include_pipeline_trace": False,
            },
        )
        if resp.status_code == 200:
            return True, str(resp.json().get("answer", ""))[:80]
        return False, f"HTTP {resp.status_code}: {resp.text[:120]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def main() -> None:
    parser = argparse.ArgumentParser(description="M4 100-question run with checkpoint/resume")
    parser.add_argument("--checkpoint", type=Path, default=Path("m4_progress.json"))
    parser.add_argument("--questions", type=Path, default=Path(QUESTIONS_PATH))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--limit", type=int, default=None, help="run at most N pending questions (trial runs)")
    parser.add_argument("--retry-failed", action="store_true", help="re-run questions recorded as failed")
    args = parser.parse_args()

    raw = json.loads(args.questions.read_text(encoding="utf-8"))
    items = raw.get("questions", raw) if isinstance(raw, dict) else raw
    low, high = raw.get("document_id_range", [9801, 9870]) if isinstance(raw, dict) else [9801, 9870]
    document_ids = list(range(low, high + 1))

    cp = load_checkpoint(args.checkpoint)
    done_ids = {qid for qid, rec in cp.items() if rec.get("status") == "ok"}
    failed_ids = {qid for qid, rec in cp.items() if rec.get("status") == "failed"}

    if args.retry_failed and failed_ids:
        # failed questions become pending again
        for qid in failed_ids:
            cp.pop(qid, None)
        done_ids -= failed_ids

    pending = [item for item in items if str(item["id"]) not in done_ids]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"base_url={args.base_url} questions={len(items)} "
        f"checkpoint={args.checkpoint} done={len(done_ids)} pending_now={len(pending)}"
    )

    ok_n, fail_n, skip_n = 0, 0, len(items) - len(done_ids) - len(pending)
    t_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as client:
        for idx, item in enumerate(pending, start=1):
            qid = str(item["id"])
            question = item["question"]
            t0 = time.perf_counter()
            ok, detail = False, "no attempt"
            for attempt in range(1, ATTEMPTS_PER_QUESTION + 1):
                ok, detail = await ask_once(client, args.base_url, question, document_ids)
                if ok:
                    break
                if attempt < ATTEMPTS_PER_QUESTION:
                    await asyncio.sleep(RETRY_DELAY_S)
            elapsed = time.perf_counter() - t0
            if ok:
                ok_n += 1
                cp[qid] = {"status": "ok", "elapsed_s": round(elapsed, 1), "at": time.time(), "answer_head": detail}
            else:
                fail_n += 1
                cp[qid] = {"status": "failed", "elapsed_s": round(elapsed, 1), "at": time.time(), "error": detail[:300]}
            save_checkpoint(args.checkpoint, cp)  # crash-safe: persisted after every question
            marker = "OK" if ok else "FAIL"
            print(
                f"  [{idx}/{len(pending)}] id={qid} {marker} {elapsed:.1f}s (attempt ok_n={ok_n} fail_n={fail_n})"
                f" | {detail[:80]}",
                flush=True,
            )

    total_time = time.perf_counter() - t_start
    final_done = sum(1 for r in cp.values() if r.get("status") == "ok")
    final_failed = sum(1 for r in cp.values() if r.get("status") == "failed")
    print(
        f"\nDONE: run_ok={ok_n} run_fail={fail_n} skipped={skip_n} "
        f"| checkpoint totals: ok={final_done}/{len(items)} failed={final_failed} "
        f"| total_time={total_time:.0f}s ({total_time / 60:.1f}min)"
    )
    if final_failed:
        print(f"failed ids: {sorted(int(q) for q, r in cp.items() if r.get('status') == 'failed')}")
        print(f"resume with: python {sys.argv[0]} --checkpoint {args.checkpoint} --retry-failed")
        sys.exit(1)


asyncio.run(main())
