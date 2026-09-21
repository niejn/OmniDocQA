"use client";

import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type {
  DocumentFilters,
  DocumentSet,
  FilterFacetChapter,
  FiltersFacet
} from "./types";

/* ── FilterBar: books/chapters independent checkboxes + kind + sets ─── */

export function FilterBar({
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
