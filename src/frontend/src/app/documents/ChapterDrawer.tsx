"use client";

import { useEffect, useState } from "react";
import { jsonFetch, toErrorMessage } from "@/lib/jsonFetch";
import { EvidenceCardView } from "./EvidenceCardView";
import type { ChapterChunks, FilterFacetChapter } from "./types";

/* ── chapter drawer (MM-4) ──────────────────────────────────────────── */

/** 章节浏览分页大小；请求参数与翻页兜底计算共用同一常量。 */
const CHAPTER_PAGE_SIZE = 20;

export function ChapterDrawer({
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
    const params = new URLSearchParams({ page: String(page), page_size: String(CHAPTER_PAGE_SIZE) });
    if (kind) params.set("kind", kind);
    jsonFetch<ChapterChunks>(`/api/documents/chapters/${chapter.document_id}/chunks?${params}`)
      .then((d) => {
        if (!stale) setData(d);
      })
      .catch((e: unknown) => {
        if (!stale) setError(toErrorMessage(e));
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
            第 {page} 页 / 共 {Math.max(1, Math.ceil((data?.total || 0) / (data?.page_size || CHAPTER_PAGE_SIZE)))} 页（{data?.total ?? 0} 块）
          </span>
          <button
            className="rounded border px-2 py-1 disabled:opacity-40"
            disabled={!!data && page * (data?.page_size || CHAPTER_PAGE_SIZE) >= (data?.total || 0)}
            onClick={() => setPage((p) => p + 1)}
          >
            下一页
          </button>
        </div>
      </div>
    </div>
  );
}
