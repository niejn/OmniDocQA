"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

/* ── types (mirror of backend document_api models) ─────────────────── */

type Collection = "text" | "multimodal";

interface FilterFacetChapter {
  document_id: number;
  chapter_index: number;
  chapter_label: string;
  filename: string;
  pages: number;
  chunks: { text: number; image: number };
}
interface FilterFacetBook {
  book_id: string;
  chapter_count: number;
  chunk_count: number;
  chapters: FilterFacetChapter[];
}
interface FiltersFacet {
  books: FilterFacetBook[];
  kinds: string[];
}
interface DocumentFilters {
  books?: string[];
  chapters?: number[];
  kinds?: ("text" | "image")[];
}
interface DocumentSet {
  set_id: string;
  name: string;
  kind: "filter" | "enumerated";
  filter_json?: DocumentFilters | null;
  chunk_ids?: string[] | null;
  chunk_count?: number;
  stale_chunk_count?: number;
}
interface Evidence {
  rank: number;
  chunk_id?: string;
  node_id?: string;
  kind: string;
  score: number;
  document_id?: number;
  filename?: string | null;
  title?: string | null;
  page_no?: number | null;
  text_preview: string;
  book_id?: string | null;
  chapter_label?: string | null;
  image_url?: string;
}
interface AskResult {
  trace_id: string;
  latency_ms?: number;
  collection: Collection;
  question: string;
  answer: string | null;
  evidence: Evidence[];
  counts?: Record<string, number>;
}
interface ChapterChunks {
  document_id: number;
  kind: string | null;
  page: number;
  page_size: number;
  total: number;
  chunks: Evidence[];
}

/* ── localStorage keys (selection persistence + invalidation cleanup) ─ */

const LS_FILTERS = "documents.filters.v1";
const LS_COLLECTION = "documents.collection.v1";

async function jsonFetch<T>(input: string, init?: RequestInit): Promise<T> {
  const res = await fetch(input, { ...init, headers: { "Content-Type": "application/json" } });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    /* non-JSON error body */
  }
  if (!res.ok) {
    const detail = (data as { detail?: { message?: string } | string })?.detail;
    const message =
      typeof detail === "string" ? detail : detail?.message || `请求失败 (${res.status})`;
    throw new Error(message);
  }
  return data as T;
}

/* ── FilterBar: books/chapters independent checkboxes + kind + sets ─── */

function FilterBar({
  facet,
  filters,
  onChange,
  sets,
  activeSetId,
  onSetSelect,
  onSetDelete,
  onSaveSelection,
  onOpenChapter
}: {
  facet: FiltersFacet | null;
  filters: DocumentFilters;
  onChange: (next: DocumentFilters) => void;
  sets: DocumentSet[];
  activeSetId: string | null;
  onSetSelect: (setId: string | null) => void;
  onSetDelete: (setId: string) => void;
  onSaveSelection: () => void;
  onOpenChapter: (chapter: FilterFacetChapter) => void;
}) {
  const [panelOpen, setPanelOpen] = useState(true);
  const [bookQuery, setBookQuery] = useState("");

  const books = useMemo(
    () => (facet?.books || []).filter((b) => b.book_id.toLowerCase().includes(bookQuery.toLowerCase())),
    [facet, bookQuery]
  );
  const toggleBook = (bookId: string) => {
    const has = (filters.books || []).includes(bookId);
    const books = has ? (filters.books || []).filter((b) => b !== bookId) : [...(filters.books || []), bookId];
    onChange({ ...filters, books });
  };
  const toggleChapter = (docId: number) => {
    const has = (filters.chapters || []).includes(docId);
    const chapters = has
      ? (filters.chapters || []).filter((c) => c !== docId)
      : [...(filters.chapters || []), docId];
    onChange({ ...filters, chapters });
  };
  const toggleKind = (kind: "text" | "image") => {
    const has = (filters.kinds || []).includes(kind);
    const kinds = has ? (filters.kinds || []).filter((k) => k !== kind) : [...(filters.kinds || []), kind];
    onChange({ ...filters, kinds });
  };

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>筛选器（书 / 章独立勾选，非级联）</CardTitle>
        <div className="flex items-center gap-2">
          {activeSetId && (
            <Badge className="border-sky-300 bg-sky-50 text-sky-700">
              集合生效
            </Badge>
          )}
          <button className="text-xs text-zinc-500 underline" onClick={() => setPanelOpen((v) => !v)}>
            {panelOpen ? "收起" : "展开"}
          </button>
        </div>
      </CardHeader>
      {panelOpen && (
        <CardContent className="space-y-4">
          {/* kind: 检索下推维度 */}
          <div className="flex items-center gap-3 text-sm">
            <span className="w-14 text-zinc-500">类型</span>
            {(["text", "image"] as const).map((k) => (
              <label key={k} className="flex items-center gap-1">
                <input type="checkbox" checked={(filters.kinds || []).includes(k)} onChange={() => toggleKind(k)} />
                {k === "text" ? "文本块" : "图片块"}
              </label>
            ))}
          </div>
          {/* set dropdown */}
          <div className="flex items-center gap-3 text-sm">
            <span className="w-14 text-zinc-500">集合</span>
            <select
              className="rounded border border-zinc-300 px-2 py-1 text-sm"
              value={activeSetId || ""}
              onChange={(e) => onSetSelect(e.target.value || null)}
            >
              <option value="">不使用集合</option>
              {sets.map((s) => (
                <option key={s.set_id} value={s.set_id}>
                  {s.name}（{s.kind === "filter" ? "filter" : `${s.chunk_count ?? "-"} 块`}）
                </option>
              ))}
            </select>
            {activeSetId && (
              <button className="text-xs text-red-500 underline" onClick={() => onSetDelete(activeSetId)}>
                删除该集合
              </button>
            )}
            <button className="text-xs text-sky-600 underline" onClick={onSaveSelection}>
              存当前勾选为集合
            </button>
          </div>
          {/* books + chapters */}
          <div>
            <input
              className="mb-2 w-full rounded border border-zinc-300 px-2 py-1 text-sm"
              placeholder="搜索书名…"
              value={bookQuery}
              onChange={(e) => setBookQuery(e.target.value)}
            />
            <div className="max-h-72 space-y-3 overflow-auto">
              {books.map((book) => (
                <div key={book.book_id} className="rounded border border-zinc-200 p-2">
                  <label className="flex items-center gap-2 text-sm font-medium">
                    <input
                      type="checkbox"
                      checked={(filters.books || []).includes(book.book_id)}
                      onChange={() => toggleBook(book.book_id)}
                    />
                    《{book.book_id}》
                    <span className="text-xs text-zinc-400">
                      {book.chapter_count} 章 · {book.chunk_count} 块
                    </span>
                  </label>
                  <div className="ml-6 mt-1 space-y-1">
                    {book.chapters.map((chapter) => (
                      <div key={chapter.document_id} className="flex items-center gap-2 text-xs">
                        <label className="flex items-center gap-1">
                          <input
                            type="checkbox"
                            checked={(filters.chapters || []).includes(chapter.document_id)}
                            onChange={() => toggleChapter(chapter.document_id)}
                          />
                          第{chapter.chapter_index}章 · {chapter.filename}
                        </label>
                        <span className="text-zinc-400">
                          {chapter.chunks.text}T/{chapter.chunks.image}I
                        </span>
                        <button
                          className="text-sky-600 underline"
                          onClick={() => onOpenChapter(chapter)}
                        >
                          浏览
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
              {!books.length && <p className="text-sm text-zinc-400">暂无书籍；请先执行多模态入库。</p>}
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  );
}

/* ── evidence card: text / image + selection checkbox ───────────────── */

function EvidenceCardView({
  evidence,
  selected,
  onToggle
}: {
  evidence: Evidence;
  selected: boolean;
  onToggle: () => void;
}) {
  const isImage = evidence.kind === "image";
  return (
    <div
      className={cn(
        "rounded-md border p-3 text-sm transition-colors",
        selected ? "border-sky-400 bg-sky-50/60" : "border-zinc-200 bg-white"
      )}
    >
      <div className="mb-1 flex items-center gap-2">
        <input type="checkbox" checked={selected} onChange={onToggle} />
        <Badge>{isImage ? "图片" : "文本"}</Badge>
        <span className="text-xs text-zinc-500">
          #{evidence.rank} · score {evidence.score.toFixed(4)}
          {evidence.book_id ? ` · 《${evidence.book_id}》` : ""}
          {evidence.page_no != null ? ` · 第${evidence.page_no + 1}页` : ""}
        </span>
      </div>
      {evidence.image_url && (
        /* 同源代理，防穿越由后端 asset store 内聚 */
        /* eslint-disable-next-line @next/next/no-img-element */
        <img
          src={evidence.image_url}
          alt={evidence.title || "evidence"}
          className="mb-2 max-h-56 rounded border border-zinc-200"
        />
      )}
      <p className="whitespace-pre-wrap text-zinc-700">{evidence.text_preview.slice(0, 600)}</p>
    </div>
  );
}

/* ── chapter drawer (MM-4) ──────────────────────────────────────────── */

function ChapterDrawer({
  chapter,
  onClose,
  selectedIds,
  onToggleChunk
}: {
  chapter: FilterFacetChapter | null;
  onClose: () => void;
  selectedIds: Set<string>;
  onToggleChunk: (chunkId: string) => void;
}) {
  const [kind, setKind] = useState<"" | "text" | "image">("");
  const [page, setPage] = useState(1);
  const [data, setData] = useState<ChapterChunks | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPage(1);
  }, [chapter, kind]);

  useEffect(() => {
    if (!chapter) return;
    setError(null);
    const params = new URLSearchParams({ page: String(page), page_size: "20" });
    if (kind) params.set("kind", kind);
    jsonFetch<ChapterChunks>(`/api/documents/chapters/${chapter.document_id}/chunks?${params}`)
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, [chapter, kind, page]);

  if (!chapter) return null;
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/30" onClick={onClose}>
      <div
        className="h-full w-[560px] overflow-auto bg-white p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold">{chapter.chapter_label}</h3>
          <button className="text-xs text-zinc-500 underline" onClick={onClose}>
            关闭
          </button>
        </div>
        <div className="mb-3 flex gap-3 text-sm">
          {[
            { v: "", label: "全部" },
            { v: "text", label: "文本" },
            { v: "image", label: "图片" }
          ].map((opt) => (
            <label key={opt.v} className="flex items-center gap-1">
              <input type="radio" checked={kind === opt.v} onChange={() => setKind(opt.v as "" | "text" | "image")} />
              {opt.label}
            </label>
          ))}
        </div>
        {error && <p className="text-sm text-red-500">{error}</p>}
        <div className="space-y-2">
          {(data?.chunks || []).map((chunk) => (
            <EvidenceCardView
              key={chunk.chunk_id}
              evidence={chunk}
              selected={!!chunk.chunk_id && selectedIds.has(chunk.chunk_id)}
              onToggle={() => chunk.chunk_id && onToggleChunk(chunk.chunk_id)}
            />
          ))}
        </div>
        <div className="mt-3 flex items-center justify-between text-sm">
          <button
            className="rounded border px-2 py-1 disabled:opacity-40"
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
          >
            上一页
          </button>
          <span className="text-zinc-500">
            第 {page} 页 / 共 {Math.max(1, Math.ceil((data?.total || 0) / (data?.page_size || 20)))} 页（{data?.total ?? 0} 块）
          </span>
          <button
            className="rounded border px-2 py-1 disabled:opacity-40"
            disabled={!!data && page * (data?.page_size || 20) >= (data?.total || 0)}
            onClick={() => setPage((p) => p + 1)}
          >
            下一页
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── main page ──────────────────────────────────────────────────────── */

export default function DocumentsPage() {
  const [facet, setFacet] = useState<FiltersFacet | null>(null);
  const [collections, setCollections] = useState<{ id: string; available: boolean; points: number | null }[]>([]);
  const [collection, setCollection] = useState<Collection>("multimodal");
  const [filters, setFilters] = useState<DocumentFilters>({});
  const [sets, setSets] = useState<DocumentSet[]>([]);
  const [activeSetId, setActiveSetId] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(8);
  const [generateAnswer, setGenerateAnswer] = useState(true);
  const [result, setResult] = useState<AskResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [chapter, setChapter] = useState<FilterFacetChapter | null>(null);
  const [saveName, setSaveName] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  /* init: facet + collections + persisted filters (with invalidation cleanup) */
  useEffect(() => {
    jsonFetch<FiltersFacet>("/api/documents/filters").then((f) => {
      setFacet(f);
      const knownBooks = new Set((f.books || []).map((b) => b.book_id));
      const knownChapters = new Set((f.books || []).flatMap((b) => b.chapters.map((c) => c.document_id)));
      try {
        const raw = window.localStorage.getItem(LS_FILTERS);
        if (raw) {
          const saved = JSON.parse(raw) as DocumentFilters;
          // 失效清洗：库已删的书/章从持久化勾选中剔除
          setFilters({
            books: (saved.books || []).filter((b) => knownBooks.has(b)),
            chapters: (saved.chapters || []).filter((c) => knownChapters.has(c)),
            kinds: saved.kinds || []
          });
        }
      } catch {
        /* corrupted storage: start clean */
      }
    }).catch((e: Error) => setError(e.message));
    fetch("/api/documents/collections")
      .then((r) => r.json())
      .then((d) => setCollections(d.collections || []))
      .catch(() => undefined);
    jsonFetch<{ sets: DocumentSet[] }>("/api/documents/sets").then((d) => setSets(d.sets)).catch(() => undefined);
    const savedCollection = window.localStorage.getItem(LS_COLLECTION);
    if (savedCollection === "text" || savedCollection === "multimodal") setCollection(savedCollection);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(LS_FILTERS, JSON.stringify(filters));
  }, [filters]);
  useEffect(() => {
    window.localStorage.setItem(LS_COLLECTION, collection);
  }, [collection]);

  const ask = useCallback(async () => {
    if (!question.trim() || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const body: Record<string, unknown> = {
        question: question.trim(),
        collection,
        top_k: topK,
        generate_answer: generateAnswer
      };
      if (collection === "multimodal") {
        if (activeSetId) body.set_id = activeSetId;
        else if (filters.books?.length || filters.chapters?.length || filters.kinds?.length)
          body.filters = filters;
      }
      const data = await jsonFetch<AskResult>("/api/documents/ask", {
        method: "POST",
        body: JSON.stringify(body)
      });
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [question, busy, collection, topK, generateAnswer, filters, activeSetId]);

  const toggleSelection = useCallback((chunkId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(chunkId)) next.delete(chunkId);
      else next.add(chunkId);
      return next;
    });
  }, []);

  const saveSelectionAsSet = useCallback(async () => {
    if (!selectedIds.size) return;
    setSaveName(`选集 ${new Date().toLocaleString()}`);
  }, [selectedIds]);

  const confirmSaveSet = useCallback(async () => {
    if (!saveName?.trim()) return;
    try {
      await jsonFetch("/api/documents/sets", {
        method: "POST",
        body: JSON.stringify({ name: saveName.trim(), chunk_ids: Array.from(selectedIds) })
      });
      setNotice(`已保存集合「${saveName.trim()}」（${selectedIds.size} 块）`);
      setSaveName(null);
      const d = await jsonFetch<{ sets: DocumentSet[] }>("/api/documents/sets");
      setSets(d.sets);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [saveName, selectedIds]);

  const mmAvailable = collections.find((c) => c.id === "multimodal")?.available;
  const evidence = result?.evidence || [];

  return (
    <main className="mx-auto max-w-6xl px-6 py-8">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">多模态文档库</h1>
          <p className="text-sm text-zinc-500">PDF 书籍入库 · 图文证据检索 · 筛选与集合策展</p>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <Link className="text-sky-600 underline" href="/">
            财务问答
          </Link>
        </div>
      </header>

      {error && (
        <div className="mb-4 rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}
      {notice && (
        <div className="mb-4 rounded border border-emerald-300 bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
          {notice}
        </div>
      )}

      <div className="grid grid-cols-[1fr_320px] gap-6">
        <section className="space-y-4">
          {/* controls */}
          <Card>
            <CardContent className="space-y-3 pt-5">
              <div className="flex items-center gap-3 text-sm">
                <span className="text-zinc-500">检索库</span>
                {(["multimodal", "text"] as const).map((c) => {
                  const info = collections.find((x) => x.id === c);
                  const disabled = c === "multimodal" && !mmAvailable;
                  return (
                    <label key={c} className={cn("flex items-center gap-1", disabled && "opacity-40")}>
                      <input
                        type="radio"
                        checked={collection === c}
                        disabled={disabled}
                        onChange={() => setCollection(c)}
                      />
                      {c === "multimodal" ? `多模态（${info?.points ?? 0} 块）` : "财务文本（SEC）"}
                    </label>
                  );
                })}
              </div>
              <div className="flex gap-2">
                <input
                  className="flex-1 rounded border border-zinc-300 px-3 py-2 text-sm"
                  placeholder={
                    collection === "multimodal" ? "例如：有界流和无界流的定义" : "例如：Why did iPhone net sales increase?"
                  }
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && ask()}
                />
                <Button onClick={ask} disabled={busy || !question.trim()}>
                  {busy ? "检索中…" : "查询"}
                </Button>
              </div>
              <div className="flex items-center gap-4 text-sm text-zinc-600">
                <label className="flex items-center gap-1">
                  top_k
                  <input
                    type="number"
                    min={1}
                    max={50}
                    className="w-16 rounded border border-zinc-300 px-1 py-0.5"
                    value={topK}
                    onChange={(e) => setTopK(Math.max(1, Math.min(50, Number(e.target.value) || 8)))}
                  />
                </label>
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={generateAnswer} onChange={(e) => setGenerateAnswer(e.target.checked)} />
                  生成回答（关闭 = 仅检索证据）
                </label>
              </div>
            </CardContent>
          </Card>

          {/* answer */}
          {result?.answer && (
            <Card>
              <CardHeader>
                <CardTitle>回答</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="whitespace-pre-wrap text-sm text-zinc-800">{result.answer}</p>
                <p className="mt-2 text-xs text-zinc-400">
                  trace {result.trace_id} · {result.latency_ms}ms · 证据 {evidence.length} 条
                </p>
              </CardContent>
            </Card>
          )}

          {/* evidence */}
          {evidence.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>证据（勾选加入集合）</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {evidence.map((e) => (
                  <EvidenceCardView
                    key={e.chunk_id || e.node_id || e.rank}
                    evidence={e}
                    selected={!!e.chunk_id && selectedIds.has(e.chunk_id)}
                    onToggle={() => e.chunk_id && toggleSelection(e.chunk_id)}
                  />
                ))}
              </CardContent>
            </Card>
          )}
        </section>

        {/* right rail: filters */}
        <aside className="space-y-4">
          <FilterBar
            facet={facet}
            filters={filters}
            onChange={setFilters}
            sets={sets}
            activeSetId={activeSetId}
            onSetSelect={(id) => {
              setActiveSetId(id);
              if (id) setFilters({}); // set 与手选 filters 互斥，UI 上选择集合即清空手选
            }}
            onSetDelete={async (id) => {
              if (!confirm("确认删除该集合？")) return;
              await fetch(`/api/documents/sets/${id}`, { method: "DELETE" });
              setActiveSetId(null);
              const d = await jsonFetch<{ sets: DocumentSet[] }>("/api/documents/sets");
              setSets(d.sets);
            }}
            onSaveSelection={saveSelectionAsSet}
            onOpenChapter={(ch) => setChapter(ch)}
          />
          {collection === "text" && (
            <p className="text-xs text-zinc-400">text 库为 SEC filings，筛选器/集合不生效（多模态域专用）。</p>
          )}
        </aside>
      </div>

      {/* selection floating bar */}
      {selectedIds.size > 0 && (
        <div className="fixed bottom-6 left-1/2 z-30 flex -translate-x-1/2 items-center gap-3 rounded-full bg-zinc-900 px-5 py-2 text-sm text-white shadow-lg">
          <span>已选 {selectedIds.size} 块</span>
          <button className="underline" onClick={saveSelectionAsSet}>
            存为集合
          </button>
          <button className="underline" onClick={() => setSelectedIds(new Set())}>
            清空
          </button>
        </div>
      )}

      {/* save-set dialog */}
      {saveName !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setSaveName(null)}>
          <div className="w-96 rounded-lg bg-white p-5 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="mb-2 text-sm font-semibold">保存枚举集合（{selectedIds.size} 块）</h3>
            <input
              className="mb-3 w-full rounded border border-zinc-300 px-2 py-1 text-sm"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="集合名称"
              autoFocus
            />
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setSaveName(null)}>
                取消
              </Button>
              <Button onClick={confirmSaveSet}>保存</Button>
            </div>
          </div>
        </div>
      )}

      <ChapterDrawer
        chapter={chapter}
        onClose={() => setChapter(null)}
        selectedIds={selectedIds}
        onToggleChunk={toggleSelection}
      />
    </main>
  );
}
