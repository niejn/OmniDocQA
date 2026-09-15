#!/usr/bin/env python3
"""M4 retrieval-backend benchmark: 100-question A/B run with full checkpoint/resume.

Orchestrates, per round (opensearch baseline vs milvus candidate):
  questions  — sequential HTTP runs against /agent/api/ask/generate, per-question
               atomic checkpoint (resume skips done, --retry-failed re-runs failed);
  scoring    — drains the rag_evaluation_jobs queue via /agent/api/ask/evaluate/pending
               (queue itself is the checkpoint: pending→completed persists in PG);
  report     — aggregates faithfulness / context_precision per round from
               rag_evaluation_jobs (backend marker in metadata, time-window fallback)
               and checks the M4 gate (±2% mean delta).

Pipeline state: m4_bench/state.json — stage per round; safe to kill any time,
re-running resumes at the first unfinished stage. Server lifecycle is managed
here (uvicorn subprocess, restarted between rounds because .env is read once
at import).

Usage (repo root):
  python m4_benchmark.py                     # run/resume everything
  python m4_benchmark.py --status            # show state and exit
  python m4_benchmark.py --retry-failed      # also re-run failed questions
  python m4_benchmark.py --only opensearch   # single round (questions+scoring)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent
AGENT_DIR = REPO_ROOT / "src" / "agent"
BENCH_DIR = REPO_ROOT / "m4_bench"
QUESTIONS_PATH = AGENT_DIR / "tools" / "data" / "apple_narrative_questions_100.json"
ENV_PATH = AGENT_DIR / ".env"
STATE_PATH = BENCH_DIR / "state.json"
SERVER_PID_PATH = BENCH_DIR / "server.pid"
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

BASE_URL = "http://127.0.0.1:8000"
HEALTH_PATH = "/agent/health"
ASK_PATH = "/agent/api/ask/generate"
EVALUATE_PATH = "/agent/api/ask/evaluate/pending"
ASK_TIMEOUT_S = 180.0
ATTEMPTS_PER_QUESTION = 3
RETRY_DELAY_S = 5.0
GATE_DELTA = 0.02  # M4 acceptance: mean metric delta within ±2%

ROUNDS = ["opensearch", "milvus"]  # baseline first, candidate second

log = logging.getLogger("m4bench")


# ---------------- persistence ----------------

def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:
            log.warning("checkpoint %s unreadable (%s); starting fresh", path, exc)
    return {}


def new_state() -> dict:
    return {"rounds": {name: {"stage": "questions"} for name in ROUNDS}, "report_done": False}


def save_state(state: dict) -> None:
    _atomic_write_json(STATE_PATH, state)


# ---------------- server lifecycle ----------------

def _kill_port_8000() -> None:
    """Kill whatever listens on :8000 (Windows netstat/taskkill)."""
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception as exc:
        log.warning("netstat failed: %s", exc)
        return
    pids = set()
    for line in out.splitlines():
        if ":8000" in line and "LISTENING" in line.upper():
            parts = line.split()
            if parts:
                pids.add(parts[-1])
    for pid in pids:
        log.info("killing stale process on :8000 (pid=%s)", pid)
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, timeout=15)


def set_env_sparse_backend(backend: str) -> None:
    """Rewrite SPARSE_BACKEND= in src/agent/.env (load_dotenv(override=True) beats process env)."""
    text = ENV_PATH.read_text(encoding="utf-8")
    new_text, n = re.subn(r"(?m)^SPARSE_BACKEND=\S.*$", f"SPARSE_BACKEND={backend}", text)
    if n == 0:
        new_text = text + f"\nSPARSE_BACKEND={backend}\n"
    if new_text != text:
        ENV_PATH.write_text(new_text, encoding="utf-8")
    log.info(".env SPARSE_BACKEND -> %s (%d line rewritten)", backend, n)


def spawn_server() -> subprocess.Popen:
    _kill_port_8000()
    proc = subprocess.Popen(
        [str(VENV_PYTHON), "-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(AGENT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    SERVER_PID_PATH.write_text(str(proc.pid), encoding="utf-8")
    log.info("spawned uvicorn pid=%s", proc.pid)
    return proc


async def wait_server_ready(timeout_s: float = 120.0, expect_backend: str | None = None) -> bool:
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        while time.perf_counter() - t0 < timeout_s:
            try:
                r = await client.get(f"{BASE_URL}{HEALTH_PATH}")
                if r.status_code == 200:
                    log.info("server ready after %.1fs", time.perf_counter() - t0)
                    return True
            except Exception:
                pass
            await asyncio.sleep(2.0)
    return False


def stop_server() -> None:
    if SERVER_PID_PATH.exists():
        try:
            pid = int(SERVER_PID_PATH.read_text().strip())
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=15)
            log.info("stopped server pid=%s", pid)
        except Exception as exc:
            log.warning("stop server failed: %s", exc)
        SERVER_PID_PATH.unlink(missing_ok=True)


# ---------------- questions stage ----------------

async def ask_once(client: httpx.AsyncClient, question: str, document_ids: list[int]) -> tuple[bool, str]:
    try:
        r = await client.post(
            f"{BASE_URL}{ASK_PATH}",
            json={
                "question": question,
                "document_ids": document_ids,
                "top_k": 3,
                "detail_level": "detailed",
                "include_pipeline_trace": False,
            },
        )
        if r.status_code == 200:
            return True, str(r.json().get("answer", ""))[:80]
        return False, f"HTTP {r.status_code}: {r.text[:120]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def run_questions(round_name: str, cp_path: Path, items: list[dict], document_ids: list[int],
                        retry_failed: bool) -> bool:
    cp = load_json(cp_path)
    done = {q for q, r in cp.items() if r.get("status") == "ok"}
    if retry_failed:
        for q in [q for q, r in cp.items() if r.get("status") == "failed"]:
            cp.pop(q, None)
        done = {q for q, r in cp.items() if r.get("status") == "ok"}
    pending = [it for it in items if str(it["id"]) not in done]
    log.info("[questions:%s] checkpoint ok=%d/%d pending=%d", round_name, len(done), len(items), len(pending))
    if not pending:
        return True

    ok_n = fail_n = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(ASK_TIMEOUT_S)) as client:
        for idx, item in enumerate(pending, start=1):
            qid = str(item["id"])
            t0 = time.perf_counter()
            ok, detail = False, "no attempt"
            for attempt in range(1, ATTEMPTS_PER_QUESTION + 1):
                ok, detail = await ask_once(client, item["question"], document_ids)
                if ok:
                    break
                if attempt < ATTEMPTS_PER_QUESTION:
                    await asyncio.sleep(RETRY_DELAY_S)
            elapsed = round(time.perf_counter() - t0, 1)
            if ok:
                ok_n += 1
                cp[qid] = {"status": "ok", "elapsed_s": elapsed, "at": time.time(), "answer_head": detail}
            else:
                fail_n += 1
                cp[qid] = {"status": "failed", "elapsed_s": elapsed, "at": time.time(), "error": detail[:300]}
            _atomic_write_json(cp_path, cp)
            log.info("[questions:%s] %d/%d id=%s %s %.1fs | %s",
                     round_name, idx, len(pending), qid, "OK" if ok else "FAIL", elapsed, detail[:70])
    log.info("[questions:%s] batch done ok=%d fail=%d (checkpoint totals ok=%d failed=%d)",
             round_name, ok_n, fail_n,
             sum(1 for r in cp.values() if r.get("status") == "ok"),
             sum(1 for r in cp.values() if r.get("status") == "failed"))
    return fail_n == 0


# ---------------- scoring stage ----------------

async def fetch_pool():
    sys.path.insert(0, str(AGENT_DIR))
    from tools.node_repository import get_pool
    return await get_pool()


async def window_job_counts(start_ts: float, end_ts: float) -> dict:
    pool = await fetch_pool()
    from datetime import datetime as dt
    rows = await pool.fetch(
        """
        SELECT status, count(*) AS n
        FROM rag_evaluation_jobs
        WHERE created_at >= $1::timestamptz AND created_at < $2::timestamptz
        GROUP BY status
        """,
        dt.fromtimestamp(start_ts, tz=UTC),
        dt.fromtimestamp(end_ts, tz=UTC),
    )
    return {r["status"]: int(r["n"]) for r in rows}


async def drain_scoring(round_name: str, start_ts: float, end_ts: float, max_polls: int = 200) -> bool:
    """POST evaluate/pending until the round window has no pending jobs left."""
    poll = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0)) as client:
        while poll < max_polls:
            counts = await window_job_counts(start_ts, end_ts)
            pending_n = counts.get("pending", 0)
            log.info("[scoring:%s] window jobs: %s", round_name, counts)
            if pending_n == 0:
                log.info("[scoring:%s] window drained (no pending).", round_name)
                return True
            try:
                r = await client.post(f"{BASE_URL}{EVALUATE_PATH}")
                body = r.json() if r.status_code < 400 else {"error": r.text[:200]}
                log.info("[scoring:%s] evaluate/pending -> %s %s", round_name, r.status_code,
                         body if isinstance(body, dict) else str(body)[:120])
            except Exception as exc:
                log.warning("[scoring:%s] evaluate call failed: %s", round_name, exc)
            poll += 1
            await asyncio.sleep(2.0)
    log.error("[scoring:%s] exceeded %d polls", round_name, max_polls)
    return False


# ---------------- report stage ----------------

async def build_report(rounds_meta: dict) -> str:
    pool = await fetch_pool()
    from datetime import datetime as dt
    lines = ["# M4 benchmark report", "", f"generated: {datetime.now().isoformat(timespec='seconds')}", ""]
    summary = {}
    # Round attribution: metadata.sparse_backend marker (jobs enqueued by the new
    # server code). Legacy no-marker jobs all belong to the first round (it ran
    # before any other backend was active), bounded by the second round's start.
    round_b_start = max((m["start"] for n, m in rounds_meta.items() if n != ROUNDS[0]), default=None)
    for name, meta in rounds_meta.items():
        rows = await pool.fetch(
            """
            SELECT
              count(*) FILTER (WHERE status = 'completed') AS completed,
              count(*) FILTER (WHERE status = 'failed')    AS failed,
              count(*) FILTER (WHERE status = 'skipped')   AS skipped,
              count(*) FILTER (WHERE status = 'pending')   AS pending,
              avg((metadata->>'faithfulness')::float)      FILTER (WHERE status = 'completed') AS faith,
              avg((metadata->>'context_precision')::float) FILTER (WHERE status = 'completed') AS ctxp,
              min((metadata->>'faithfulness')::float)      FILTER (WHERE status = 'completed') AS faith_min,
              max((metadata->>'faithfulness')::float)      FILTER (WHERE status = 'completed') AS faith_max
            FROM rag_evaluation_jobs
            WHERE created_at >= $1::timestamptz
              AND created_at < $2::timestamptz
              AND (
                    metadata->>'sparse_backend' = $3
                    OR (
                        metadata->>'sparse_backend' IS NULL
                        AND $3 = $4  -- legacy jobs only for the first round
                        AND ($5::timestamptz IS NULL OR created_at < $5::timestamptz)
                    )
                  )
            """,
            dt.fromtimestamp(meta["start"], tz=UTC),
            dt.fromtimestamp(meta["end"], tz=UTC),
            name,
            ROUNDS[0],
            dt.fromtimestamp(round_b_start, tz=UTC) if round_b_start else None,
        )
        r = rows[0]
        summary[name] = {
            "completed": r["completed"], "failed": r["failed"], "skipped": r["skipped"],
            "pending": r["pending"],
            "faithfulness": r["faith"], "context_precision": r["ctxp"],
        }
        lines.append(
            f"- {name}: completed={r['completed']} failed={r['failed']} skipped={r['skipped']} "
            f"pending={r['pending']} | faithfulness={_fmt(r['faith'])} "
            f"(min {_fmt(r['faith_min'])} max {_fmt(r['faith_max'])}) | context_precision={_fmt(r['ctxp'])}"
        )
    lines += ["", "| round | completed | faithfulness | context_precision |", "|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['completed']} | {_fmt(s['faithfulness'])} | {_fmt(s['context_precision'])} |")

    if len(summary) == 2:
        a, b = summary[ROUNDS[0]], summary[ROUNDS[1]]
        lines += ["", "## M4 gate (mean delta vs baseline, tolerance ±2%)", ""]
        for metric in ("faithfulness", "context_precision"):
            va, vb = a.get(metric), b.get(metric)
            if va is None or vb is None:
                lines.append(f"- {metric}: missing data (baseline={va} candidate={vb}) — gate NOT passed")
                continue
            delta = vb - va
            verdict = "PASS" if abs(delta) <= GATE_DELTA else "FAIL"
            lines.append(f"- {metric}: baseline={va:.4f} candidate={vb:.4f} delta={delta:+.4f} "
                         f"({delta / va * 100:+.2f}%) → {verdict}")
    report = "\n".join(lines)
    (BENCH_DIR / "report.md").write_text(report, encoding="utf-8")
    return report


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


# ---------------- orchestration ----------------

async def run(only: str | None, retry_failed: bool) -> None:
    state = load_json(STATE_PATH) or new_state()
    raw = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    items = raw.get("questions", raw) if isinstance(raw, dict) else raw
    low, high = raw.get("document_id_range", [9801, 9870]) if isinstance(raw, dict) else [9801, 9870]
    document_ids = list(range(low, high + 1))

    rounds = [only] if only else ROUNDS
    for name in rounds:
        rstate = state["rounds"].setdefault(name, {"stage": "questions"})
        cp_path = BENCH_DIR / f"round_{name}_progress.json"
        if name == "opensearch" and (REPO_ROOT / "m4_progress_roundA_os.json").exists():
            # adopt progress from the standalone runner if bench hasn't started its own
            if not cp_path.exists():
                cp_path.write_text((REPO_ROOT / "m4_progress_roundA_os.json").read_text(encoding="utf-8"),
                                   encoding="utf-8")
                log.info("adopted existing checkpoint m4_progress_roundA_os.json -> %s", cp_path)

        if rstate["stage"] == "questions":
            log.info("=== round %s / stage questions ===", name)
            rstate["start"] = rstate.get("start") or _first_checkpoint_time(cp_path) or time.time()
            set_env_sparse_backend(name)
            spawn_server()
            if not await wait_server_ready(expect_backend=name):
                log.error("server did not become ready for round %s; aborting (state saved)", name)
                save_state(state)
                stop_server()
                return
            ok = await run_questions(name, cp_path, items, document_ids, retry_failed)
            rstate.setdefault("windows", []).append({"start": rstate["start"], "end": time.time()})
            # Failed questions no longer gate the pipeline: permanently-failed
            # questions (e.g. provider SensitiveContent blocks) would otherwise
            # block scoring forever. They are counted in the report instead.
            if not ok:
                log.warning("round %s has failed questions (counted in report); continuing to scoring", name)
            rstate["stage"] = "scoring"
            stop_server()
            # scoring needs the server too — respawn once (backend config is now irrelevant
            # because scoring only reads stored jobs; keep the round's env as-is)
            spawn_server()
            if not await wait_server_ready():
                log.error("server not ready for scoring; aborting (state saved)")
                save_state(state)
                stop_server()
                return
            save_state(state)

        if rstate["stage"] == "scoring":
            log.info("=== round %s / stage scoring ===", name)
            # scoring POSTs evaluate/pending — needs a live server. Direct stage
            # entry (resume from state) skips the questions branch, so ensure it.
            # Always kill + respawn: a surviving server from a previous run may
            # predate current code/config (e.g. judge LLM fixes) — never trust it.
            spawn_server()
            if not await wait_server_ready():
                log.error("server not ready for scoring; aborting (state saved)")
                save_state(state)
                stop_server()
                return
            window_start = min((w["start"] for w in rstate.get("windows", [])), default=rstate.get("start", 0))
            window_end = max((w["end"] for w in rstate.get("windows", [])), default=time.time())
            drained = await drain_scoring(name, window_start, window_end + 3600)
            if drained:
                rstate["stage"] = "done"
            save_state(state)

        if rstate["stage"] == "done" and not only:
            continue

    stop_server()

    if not only and not state.get("report_done"):
        log.info("=== report ===")
        rounds_meta = {}
        for name in ROUNDS:
            rstate = state["rounds"].get(name, {})
            ws = rstate.get("windows") or []
            if ws:
                rounds_meta[name] = {"start": min(w["start"] for w in ws),
                                     "end": max(w["end"] for w in ws) + 3600}
        if len(rounds_meta) == len(ROUNDS):
            report = await build_report(rounds_meta)
            log.info("report written to %s/report.md\n%s", BENCH_DIR, report)
            state["report_done"] = True
        else:
            log.warning("not all rounds complete; report skipped (state: %s)",
                        {k: v.get("stage") for k, v in state["rounds"].items()})
        save_state(state)

def _first_checkpoint_time(cp_path: Path) -> float | None:
    cp = load_json(cp_path)
    times = [t for t in (r.get("at") for r in cp.values() if isinstance(r, dict)) if isinstance(t, (int, float))]
    return min(times) if times else None


def show_status() -> None:
    state = load_json(STATE_PATH)
    print(f"state: {STATE_PATH}")
    for name in ROUNDS:
        rstate = state.get("rounds", {}).get(name, {"stage": "not-started"})
        cp = load_json(BENCH_DIR / f"round_{name}_progress.json")
        ok = sum(1 for r in cp.values() if r.get("status") == "ok")
        failed = sum(1 for r in cp.values() if r.get("status") == "failed")
        print(f"  {name:12s} stage={rstate.get('stage'):10s} questions ok={ok} failed={failed}")
    print(f"  report_done={state.get('report_done')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 backend benchmark with checkpoint/resume")
    parser.add_argument("--only", choices=ROUNDS, default=None, help="run a single round")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--status", action="store_true", help="print state and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    BENCH_DIR.mkdir(exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout)]
    if not args.status:
        log_path = BENCH_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        print(f"log: {log_path}")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    if args.status:
        show_status()
        return
    asyncio.run(run(args.only, args.retry_failed))


if __name__ == "__main__":
    main()
