"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { jsonFetch } from "./jsonFetch";
import { collectionKind } from "./types";
import type {
  CollectionInfo,
  EvalSummarySection,
  StartJobResponse,
  TestsetJobStatus,
  TestsetQuestion
} from "./types";

/* ── eval panel: RAGAS testset generation + evaluation (202 job + polling) ── */

const POLL_INTERVAL_MS = 3000;
// P2-2: 100 题 LLM 生成实测可能超 3 分钟，轮询上限从 60 次（约 180s）放宽到 200 次（约 600s）。
const MAX_POLLS = 200;

type JobKind = "generate" | "evaluate";

/** questions 条目兼容纯文本或对象，取题目文本；未知结构兜底为 JSON 串。 */
function questionText(q: TestsetQuestion): string {
  if (typeof q === "string") return q;
  return q.question || q.text || q.question_text || JSON.stringify(q);
}

function formatMetric(value: unknown): string {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (value === null || value === undefined) return "-";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/* ── P2-4: summary 渲染 ──────────────────────────────────────────────────
   后端 summary 是嵌套结构：{ overall | <qtype>: { n, errors, hit_rate,
   gold_recall, mrr, scope_violation?, p95_ms? } }。按小节展开为人类可读的
   指标行；未知嵌套值经 formatMetric 的 JSON 兜底展示，不再整块 stringify。 */

const SECTION_LABELS: Record<string, string> = { overall: "整体" };

const METRIC_LABELS: Record<string, string> = {
  n: "题数",
  errors: "错误数",
  hit_rate: "HitRate",
  gold_recall: "GoldRecall",
  mrr: "MRR",
  scope_violation: "ScopeViolation",
  p95_ms: "P95 耗时(ms)"
};

function SummaryView({ summary }: { summary: Record<string, unknown> }) {
  return (
    <div className="space-y-1 rounded border border-emerald-200 bg-emerald-50/60 p-2">
      <p className="text-xs font-medium text-emerald-700">评测指标（summary）</p>
      {Object.entries(summary).map(([key, value]) => {
        // 小节（overall / qtype 分组）：值为对象 → 展开为指标行
        if (value && typeof value === "object" && !Array.isArray(value)) {
          const rows = Object.entries(value as EvalSummarySection);
          return (
            <div key={key} className="rounded border border-emerald-100 bg-white/70 p-1.5">
              <p className="text-xs font-medium text-emerald-700">{SECTION_LABELS[key] ?? key}</p>
              {rows.map(([metricKey, metricValue]) => (
                <p key={metricKey} className="text-xs text-zinc-700">
                  {METRIC_LABELS[metricKey] ?? metricKey}: {formatMetric(metricValue)}
                </p>
              ))}
            </div>
          );
        }
        // 兜底：summary 顶层直接是标量（旧格式）→ 原样作为指标行
        return (
          <p key={key} className="text-xs text-zinc-700">
            {METRIC_LABELS[key] ?? key}: {formatMetric(value)}
          </p>
        );
      })}
    </div>
  );
}

export function EvalPanel({ collections }: { collections: CollectionInfo[] }) {
  const [open, setOpen] = useState(false);
  const [collection, setCollection] = useState("multimodal");
  const [size, setSize] = useState(10);
  const [genRunning, setGenRunning] = useState(false);
  const [evalRunning, setEvalRunning] = useState(false);
  const [testset, setTestset] = useState<{
    count: number | null;
    previews: string[];
    testsetPath: string | null;
    /** P1-2: 生成时所在的集合；与当前选择不一致则视为错配，禁止运行评测。 */
    collection: string;
  } | null>(null);
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);
  const [metricsRaw, setMetricsRaw] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const seqRef = useRef(0);
  const pollsRef = useRef(0);

  const stopPolling = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // 组件卸载时停止轮询，防止悬挂 timer
  useEffect(() => stopPolling, [stopPolling]);

  // P2-3: generate/evaluate 只接受多模态域集合——text 是 SEC filings 纯文本库，
  // 后端这两个接口对它必然 422，直接从选项剔除。
  // 规则：先按 id !== "text"（兼容不返回 kind 字段的旧后端）；响应带 kind 时再按
  // 「动态库或 multimodal」收窄，防止未来新增其它 fixed 库混入。
  const evalCollections = useMemo(
    () =>
      collections.filter((c) => {
        if (c.id === "text") return false;
        return collectionKind(c) !== "fixed" || c.id === "multimodal";
      }),
    [collections]
  );

  // 过滤后当前选中库不在选项内（如 text 被剔除、列表降级）→ 回退第一个选项，防 select 空显
  useEffect(() => {
    if (evalCollections.length > 0 && !evalCollections.some((c) => c.id === collection)) {
      setCollection(evalCollections[0].id);
    }
  }, [evalCollections, collection]);

  const pollJob = useCallback(
    (jobId: string, kind: JobKind, onDone: (d: TestsetJobStatus) => void) => {
      stopPolling();
      // P2-1: 代际 token——每次启动新轮询递增 seq；过期代际的响应一律丢弃，
      // 防止已被作废（如切换集合）或已被取代的旧任务响应覆盖新状态。
      const seq = ++seqRef.current;
      const isStale = () => seq !== seqRef.current;
      pollsRef.current = 0;
      const settle = () => {
        stopPolling();
        if (isStale()) return; // 旧代际不得触碰 running 状态（新任务可能已在跑）
        if (kind === "generate") setGenRunning(false);
        else setEvalRunning(false);
      };
      // 轮询方式：链式 setTimeout（上一次请求 settle 后再排下一次），替代 setInterval，
      // 避免慢响应期间请求堆积/并发打点；每一拍响应都先过代际校验再落地。
      const tick = () => {
        if (isStale()) return; // 已被新代际取代：静默退出，不再发请求
        timerRef.current = setTimeout(() => {
          void (async () => {
            if (isStale()) return;
            pollsRef.current += 1;
            if (pollsRef.current > MAX_POLLS) {
              settle();
              setError(
                `任务轮询超过 ${Math.round((MAX_POLLS * POLL_INTERVAL_MS) / 60000)} 分钟（${MAX_POLLS} 次）仍未完成，已停止；可稍后重试。`
              );
              return;
            }
            try {
              const d = await jsonFetch<TestsetJobStatus>(`/api/documents/testset/${encodeURIComponent(jobId)}`);
              if (isStale()) return; // P2-1: 迟到响应，丢弃
              if (d.status === "running") {
                tick();
                return;
              }
              settle();
              onDone(d);
            } catch (e) {
              if (isStale()) return;
              settle();
              setError(e instanceof Error ? e.message : String(e));
            }
          })();
        }, POLL_INTERVAL_MS);
      };
      tick();
    },
    [stopPolling]
  );

  // P1-2: 切换集合时的失效语义——
  // 1) 有在途任务（生成/评测）：后端无取消接口，做前端「标记失效」——递增代际 seq 并停止
  //    轮询，在途任务的一切迟到响应被丢弃，running 状态复位，可立即在新集合重新操作。
  // 2) 已有评测集：不删除，按 testset.collection !== collection 派生「错配」状态，
  //    禁用「运行评测」并提示；切回原集合即恢复可用（评测集数据本身未失效）。
  const handleCollectionChange = (next: string) => {
    if (next === collection) return;
    setCollection(next);
    if (genRunning || evalRunning) {
      seqRef.current += 1;
      stopPolling();
      setGenRunning(false);
      setEvalRunning(false);
      setInfo("已切换集合，原任务已作废；请在新集合下重新生成/评测。");
    }
  };

  const generate = useCallback(async () => {
    if (genRunning || evalRunning) return;
    setError(null);
    setMetrics(null);
    setMetricsRaw(null);
    setInfo("评测题生成中…（每 3s 轮询任务状态）");
    setGenRunning(true);
    try {
      const started = await jsonFetch<StartJobResponse>("/api/documents/generate-testset", {
        method: "POST",
        body: JSON.stringify({ collection, testset_size: size })
      });
      pollJob(started.job_id, "generate", (d) => {
        if (d.status === "failed") {
          setError(d.error || "评测题生成失败");
          return;
        }
        const questions = (d.questions || []).map(questionText);
        // P1-2: 记录生成时所在集合；运行期切换集合会作废本任务（seq 递增），
        // 因此该回调触发时 collection 必然仍是发起时的值。
        setTestset({
          count: d.questions_count ?? questions.length,
          previews: questions.slice(0, 3),
          testsetPath: d.testset_path || null,
          collection
        });
        setInfo(`生成完成：${d.questions_count ?? questions.length} 题`);
      });
    } catch (e) {
      setGenRunning(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [genRunning, evalRunning, collection, size, pollJob]);

  const runEvaluate = useCallback(async () => {
    if (genRunning || evalRunning) return;
    if (!testset?.testsetPath) {
      setError("缺少 testset_path：生成任务的完成响应未返回该字段，无法运行评测。");
      return;
    }
    if (testset.collection !== collection) {
      // P1-2: 双保险——按钮已禁用，这里兜底拦截编程路径/竞态下的跨集合评测
      setError(`评测集属于集合 ${testset.collection}，当前集合为 ${collection}，请切回或重新生成。`);
      return;
    }
    setError(null);
    setMetrics(null);
    setMetricsRaw(null);
    setInfo("评测运行中…（每 3s 轮询任务状态）");
    setEvalRunning(true);
    try {
      const started = await jsonFetch<StartJobResponse>("/api/documents/evaluate", {
        method: "POST",
        body: JSON.stringify({ collection, testset_path: testset.testsetPath })
      });
      pollJob(started.job_id, "evaluate", (d) => {
        if (d.status === "failed") {
          setError(d.error || "评测运行失败");
          return;
        }
        const summary = d.report?.summary;
        if (summary && Object.keys(summary).length > 0) {
          setMetrics(summary);
          setInfo("评测完成");
        } else if (d.report) {
          setMetricsRaw(JSON.stringify(d.report, null, 2).slice(0, 800));
          setInfo("评测完成（报告无 summary 字段，展示原始 JSON 摘要）");
        } else {
          setInfo("评测完成（响应未包含报告内容）");
        }
      });
    } catch (e) {
      setEvalRunning(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [genRunning, evalRunning, collection, testset, pollJob]);

  const running = genRunning || evalRunning;
  // P1-2: 评测集归属集合与当前选择不一致 → 错配：禁用「运行评测」并提示
  const testsetMismatch = !!testset && testset.collection !== collection;

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>评测（评测题生成 / 运行评测）</CardTitle>
        <button className="text-xs text-zinc-500 underline" onClick={() => setOpen((v) => !v)}>
          {open ? "收起" : "展开"}
        </button>
      </CardHeader>
      {open && (
        <CardContent className="space-y-3 text-sm">
          <div className="flex items-center gap-2">
            <span className="shrink-0 text-zinc-500">集合</span>
            <select
              className="min-w-0 flex-1 rounded border border-zinc-300 px-2 py-1 text-sm"
              value={collection}
              /* 任务运行期间也允许切换集合：切换即作废在途任务（见 handleCollectionChange），
                 给用户一条逃离长任务的出路，同时杜绝"换库还拿旧任务结果"的错配。 */
              onChange={(e) => handleCollectionChange(e.target.value)}
            >
              {evalCollections.map((c) => (
                <option key={c.id} value={c.id} disabled={!c.available}>
                  {c.id}
                  {c.available ? "" : "（不可用）"}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <span className="shrink-0 text-zinc-500">题数</span>
            <input
              type="number"
              min={1}
              max={100}
              className="w-20 rounded border border-zinc-300 px-2 py-1 text-sm"
              value={size}
              disabled={running}
              onChange={(e) => setSize(Math.max(1, Math.min(100, Number(e.target.value) || 10)))}
            />
            <Button size="sm" disabled={running} onClick={() => void generate()}>
              {genRunning ? "生成中…" : "生成评测题"}
            </Button>
          </div>
          {error && <p className="text-xs text-red-600">{error}</p>}
          {info && !error && <p className="text-xs text-zinc-500">{info}</p>}
          {testset && (
            <div className="space-y-1 rounded border border-zinc-200 p-2">
              <p className="text-xs font-medium text-zinc-700">
                评测集（{testset.collection}）：共 {testset.count ?? "?"} 题
              </p>
              {testsetMismatch && (
                <p className="text-xs text-amber-600">
                  评测集属于集合 {testset.collection}，当前集合为 {collection}，请切回或重新生成。
                </p>
              )}
              {testset.previews.map((q, i) => (
                <p key={i} className="truncate text-xs text-zinc-500">
                  {i + 1}. {q}
                </p>
              ))}
              {testset.testsetPath && (
                <p className="truncate text-xs text-zinc-400" title={testset.testsetPath}>
                  testset_path: {testset.testsetPath}
                </p>
              )}
              <Button
                size="sm"
                variant="outline"
                disabled={running || !testset.testsetPath || testsetMismatch}
                onClick={() => void runEvaluate()}
              >
                {evalRunning ? "评测中…" : "运行评测"}
              </Button>
              {!testset.testsetPath && (
                <p className="text-xs text-zinc-400">后端未返回 testset_path，暂无法运行评测。</p>
              )}
            </div>
          )}
          {metrics && <SummaryView summary={metrics} />}
          {metricsRaw && (
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded border border-zinc-200 bg-zinc-50 p-2 text-xs text-zinc-600">
              {metricsRaw}
            </pre>
          )}
        </CardContent>
      )}
    </Card>
  );
}
