"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { DocumentList } from "./DocumentList";
import { EvalPanel } from "./EvalPanel";
import { jsonFetch } from "./jsonFetch";
import { UploadPanel } from "./UploadPanel";
import type {
  AskResult,
  ChapterChunks,
  Collection,
  CollectionInfo,
  CollectionsResponse,
  CreateCollectionPayload,
  DocumentFilters,
  DocumentListItem,
  DocumentSet,
  Evidence,
  FilterFacetChapter,
  FiltersFacet,
  UploadCollectionOption,
  UploadResult
} from "./types";
import { collectionKind } from "./types";

/* ── localStorage keys (selection persistence + invalidation cleanup) ─ */

const LS_FILTERS = "documents.filters.v1";
const LS_COLLECTION = "documents.collection.v1";

/* collections 接口不可达时的兜底固定库，保持旧行为（multimodal/text 可见可选）。 */
const FALLBACK_COLLECTIONS: CollectionInfo[] = [
  { id: "multimodal", available: true, points: null, kind: "fixed" },
  { id: "text", available: true, points: null, kind: "fixed" }
];

function collectionLabel(id: string): string {
  if (id === "multimodal") return "多模态";
  if (id === "text") return "财务文本（SEC）";
  return id;
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

  // 切章时立即清空旧数据，避免慢响应期间闪现上一章内容
  useEffect(() => {
    setData(null);
  }, [chapter]);

  useEffect(() => {
    if (!chapter) return;
    let stale = false; // 竞态守卫：翻页/切章后旧响应不得覆盖新状态
    setError(null);
    const params = new URLSearchParams({ page: String(page), page_size: "20" });
    if (kind) params.set("kind", kind);
    jsonFetch<ChapterChunks>(`/api/documents/chapters/${chapter.document_id}/chunks?${params}`)
      .then((d) => {
        if (!stale) setData(d);
      })
      .catch((e: Error) => {
        if (!stale) setError(e.message);
      });
    return () => {
      stale = true;
    };
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
        {!data && !error && <p className="text-sm text-zinc-400">加载中…</p>}
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
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [collection, setCollection] = useState<Collection>("multimodal");
  const [uploadCollection, setUploadCollection] = useState("multimodal");
  const [creatingCollection, setCreatingCollection] = useState(false);
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
  const [hydrated, setHydrated] = useState(false);
  // P1-1: collection 侧独立 hydration 标志。语义：仅当 collections fetch 成功且
  // "恢复持久化库"的决策已落地（恢复成功，或本地从未保存过）才置位；
  // fetch 失败 / 保存值不在列表或不可用时保持 false，LS 原值得以保留
  // （动态库恢复可用后下次进入页面仍能自动恢复）。persist effect 据此门控。
  const [collectionHydrated, setCollectionHydrated] = useState(false);
  // P1-1: 区分"用户手动切换"与"初始写入"——hydrate 完成前用户切库也应立即生效并持久化。
  const collectionTouchedRef = useRef(false);
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [deletingDocId, setDeletingDocId] = useState<number | null>(null);

  const refreshFacets = useCallback(async () => {
    const f = await jsonFetch<FiltersFacet>("/api/documents/filters");
    setFacet(f);
    return f;
  }, []);

  const refreshDocuments = useCallback(async () => {
    const list = await jsonFetch<DocumentListItem[]>("/api/documents/documents");
    setDocuments(Array.isArray(list) ? list : []);
    return list;
  }, []);

  /* init: facet + collections + persisted filters (with invalidation cleanup) */
  useEffect(() => {
    // hydrated 仅在 facet 成功恢复后置位：失败路径不触发 persist，用户 localStorage 勾选保持原值
    refreshFacets()
      .then((f) => {
        try {
          const raw = window.localStorage.getItem(LS_FILTERS);
          if (raw) {
            const saved = JSON.parse(raw) as DocumentFilters;
            // P2-6: facet.books 为空数组时跳过失效清洗，直接按保存内容恢复。
            // 原因：删光全部文档是可达状态，此时 facets 返回空 books 并不代表勾选失效；
            // 若照常清洗会把持久化勾选全部剔除，并随 persist effect 写回空值（误清 LS）。
            // 保持 LS 原值（重写为相同内容），待 facet 恢复非空后下次进入页面再清洗。
            if ((f.books || []).length > 0) {
              const knownBooks = new Set(f.books.map((b) => b.book_id));
              const knownChapters = new Set(f.books.flatMap((b) => b.chapters.map((c) => c.document_id)));
              // 失效清洗：库已删的书/章从持久化勾选中剔除
              setFilters({
                books: (saved.books || []).filter((b) => knownBooks.has(b)),
                chapters: (saved.chapters || []).filter((c) => knownChapters.has(c)),
                kinds: saved.kinds || []
              });
            } else {
              setFilters({ books: saved.books || [], chapters: saved.chapters || [], kinds: saved.kinds || [] });
            }
          }
        } catch {
          /* corrupted storage: start clean */
        }
        setHydrated(true);
      })
      .catch((e: Error) => setError(e.message));
    refreshDocuments().catch(() => undefined); // 文档列表失败静默降级，不阻塞主功能
    const savedCollection = window.localStorage.getItem(LS_COLLECTION);
    // P2-7: 裸 fetch → jsonFetch：非 2xx 也会走 catch 降级 FALLBACK，不再静默吞错。
    jsonFetch<CollectionsResponse>("/api/documents/collections")
      .then((d) => {
        const list = Array.isArray(d.collections) ? d.collections : [];
        setCollections(list);
        if (!collectionTouchedRef.current && savedCollection && list.some((c) => c.id === savedCollection && c.available)) {
          // 恢复成功：恢复值与 LS 一致，开放后续 persist 无覆盖风险
          setCollection(savedCollection);
          setCollectionHydrated(true);
        } else if (!savedCollection) {
          // 本地从未保存过：开放 persist（写入当前值无覆盖风险）
          setCollectionHydrated(true);
        }
        // 其余情况（保存值不在列表/不可用，或用户已抢先手动切换）：保持门控不置位，
        // LS 原值保留——动态库恢复 available 后下次进入页面仍能自动恢复；
        // 用户此后的手动切换经 collectionTouchedRef 直写持久化，不受门控影响。
      })
      .catch((e: Error) => {
        // 降级行为保留：collections 置空 → collectionList 回退 FALLBACK_COLLECTIONS。
        // 取舍：初始化的非阻塞路径，走红色 error 通道太吵，改用 console.warn + 顶部 notice 弱提示。
        console.warn("[documents] collections 接口不可达，已回退到固定库列表", e);
        setNotice("集合列表获取失败，已回退到固定库（multimodal / text）。");
        // 失败不置位 collectionHydrated：LS 中保存的库选择保留到下次成功 fetch
      });
    jsonFetch<{ sets: DocumentSet[] }>("/api/documents/sets").then((d) => setSets(d.sets)).catch(() => undefined);
  }, [refreshFacets, refreshDocuments]);

  useEffect(() => {
    if (!hydrated) return; // 水合前不写入，防止初始空 filters 覆盖用户持久化勾选
    window.localStorage.setItem(LS_FILTERS, JSON.stringify(filters));
  }, [filters, hydrated]);
  useEffect(() => {
    // P1-1: collection hydrate（fetch + 恢复决策）落地前不写 LS，防止初始 "multimodal"
    // 抢先覆盖保存的库选择；但用户手动切换（touched ref）不受门控，随时生效并持久化。
    if (!collectionHydrated && !collectionTouchedRef.current) return;
    window.localStorage.setItem(LS_COLLECTION, collection);
  }, [collection, collectionHydrated]);

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
      // P2-5: filters/set 是多模态域能力，后端对动态多模态库同样支持——凡非 text 库一律下推；
      // text（SEC filings 纯文本域）维持原有提示逻辑，不携带筛选参数。
      if (collection !== "text") {
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

  /* 创建动态集合：成功后刷新列表并自动选为新上传目标；失败 detail 已可见 */
  const createCollection = useCallback(
    async (name: string, description: string): Promise<boolean> => {
      setCreatingCollection(true);
      setError(null);
      setNotice(null);
      try {
        const body: CreateCollectionPayload = { name };
        if (description) body.description = description;
        await jsonFetch("/api/documents/collections", {
          method: "POST",
          body: JSON.stringify(body)
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        return false;
      } finally {
        setCreatingCollection(false);
      }
      setUploadCollection(name); // 自动选中新集合（即使随后列表刷新失败也不影响选中）
      setNotice(`已创建集合「${name}」，已选为上传目标`);
      try {
        const d = await jsonFetch<CollectionsResponse>("/api/documents/collections");
        setCollections(Array.isArray(d.collections) ? d.collections : []);
      } catch {
        /* 刷新列表失败静默：创建本身已成功，下次进入页面会拿到新列表 */
      }
      return true;
    },
    []
  );

  const handleUpload = useCallback(
    async (file: File) => {
      if (uploading) return;
      setUploading(true);
      setError(null);
      setNotice(null);
      try {
        const form = new FormData();
        form.append("file", file);
        form.append("collection", uploadCollection); // 契约要求显式携带目标集合（默认 multimodal）
        const result = await jsonFetch<UploadResult>("/api/documents/upload", {
          method: "POST",
          body: form
        });
        const statusNote =
          result.status === "completed" ? "已入库，可直接检索" : `状态 ${result.status}（未完成入库）`;
        setNotice(
          `${result.filename} → document_id ${result.document_id}，${result.page_count} 页 ${result.node_count} 块，${statusNote}`
        );
        await Promise.all([refreshFacets(), refreshDocuments()]);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setUploading(false);
      }
    },
    [uploading, uploadCollection, refreshFacets, refreshDocuments]
  );

  const handleDeleteDocument = useCallback(
    async (doc: DocumentListItem) => {
      if (deletingDocId !== null) return;
      const label = doc.title || doc.filename;
      if (!confirm(`确认删除文档「${label}」（id ${doc.document_id}）？该操作不可撤销。`)) return;
      setDeletingDocId(doc.document_id);
      setError(null);
      setNotice(null);
      try {
        await jsonFetch(`/api/documents/documents/${doc.document_id}`, { method: "DELETE" });
        setDocuments((prev) => prev.filter((d) => d.document_id !== doc.document_id));
        setNotice(`已删除文档「${label}」（id ${doc.document_id}）`);
        await refreshFacets();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setDeletingDocId(null);
      }
    },
    [deletingDocId, refreshFacets]
  );

  /* 完整库列表（fixed + dynamic）；接口失败时回退到两个固定库 */
  const collectionList = collections.length > 0 ? collections : FALLBACK_COLLECTIONS;
  /* 上传可写集合：multimodal 固定库 + kind=dynamic 且 available 的库（排除 text 等 fixed 库）。
     P2-8: multimodal available=false 时仍保留入选，但携带 available 标记供下拉文案标注。 */
  const writableCollections = useMemo<UploadCollectionOption[]>(() => {
    const opts: UploadCollectionOption[] = [];
    const mm = collectionList.find((c) => c.id === "multimodal");
    if (mm) opts.push({ id: mm.id, points: mm.points, available: mm.available });
    for (const c of collectionList) {
      if (collectionKind(c) !== "dynamic" || !c.available || c.id === "multimodal") continue;
      opts.push({ id: c.id, points: c.points, available: c.available });
    }
    return opts;
  }, [collectionList]);
  /* P2-9: 上传下拉展示列表——当前 uploadCollection 不在可写列表时（新建集合后列表刷新
     失败、库被他端删除等），追加一个临时占位选项（标注「探测中/不可用」），
     避免 select 空显与"看起来已选中实际未带上"的静默 422。 */
  const uploadOptions = useMemo<UploadCollectionOption[]>(() => {
    if (writableCollections.some((o) => o.id === uploadCollection)) return writableCollections;
    return [...writableCollections, { id: uploadCollection, points: null, available: false, pending: true }];
  }, [writableCollections, uploadCollection]);
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
                <span className="shrink-0 text-zinc-500">检索库</span>
                <select
                  className="min-w-0 flex-1 rounded border border-zinc-300 px-2 py-1 text-sm"
                  value={collection}
                  onChange={(e) => {
                    // P1-1: 标记用户操作——hydrate 完成前手动切库也立即生效并持久化
                    collectionTouchedRef.current = true;
                    setCollection(e.target.value);
                  }}
                >
                  {collectionList.map((c) => (
                    <option key={c.id} value={c.id} disabled={!c.available}>
                      {collectionLabel(c.id)}（{c.points ?? 0} 块）
                      {c.available ? "" : " · 不可用"}
                      {collectionKind(c) === "dynamic" ? " · 动态" : ""}
                    </option>
                  ))}
                </select>
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

        {/* right rail: upload + filters + inventory */}
        <aside className="space-y-4">
          <UploadPanel
            uploading={uploading}
            onUpload={(file) => void handleUpload(file)}
            onError={setError}
            collectionOptions={uploadOptions}
            collection={uploadCollection}
            onCollectionChange={(id) => {
              setUploadCollection(id);
              // P2-8: 选中不可用集合（multimodal available=false 时仍保留入选）给弱提示：
              // 该库 Qdrant collection 缺失，上传大概率 502。走 notice 弱提示而非 error 红条。
              const opt = writableCollections.find((o) => o.id === id);
              if (opt && !opt.available) {
                setNotice(`集合「${id}」当前不可用，上传可能失败（502）；请确认该库索引是否就绪。`);
              }
            }}
            onCreateCollection={createCollection}
            creatingCollection={creatingCollection}
          />
          <EvalPanel collections={collectionList} />
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
              try {
                await jsonFetch(`/api/documents/sets/${id}`, { method: "DELETE" });
              } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
                return; // 删除失败：保留 activeSetId 与列表原状，错误已可见
              }
              setActiveSetId(null);
              try {
                const d = await jsonFetch<{ sets: DocumentSet[] }>("/api/documents/sets");
                setSets(d.sets);
              } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
              }
            }}
            onSaveSelection={saveSelectionAsSet}
            onOpenChapter={(ch) => setChapter(ch)}
          />
          <DocumentList documents={documents} deletingId={deletingDocId} onDelete={(doc) => void handleDeleteDocument(doc)} />
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
