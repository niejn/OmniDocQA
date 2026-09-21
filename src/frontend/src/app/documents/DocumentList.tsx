"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DocumentListItem } from "./types";

/* ── ingested-document inventory: title/pages/chunks + delete ────────── */

export function DocumentList({
  documents,
  deletingId,
  onDelete
}: {
  documents: DocumentListItem[];
  deletingId: number | null;
  onDelete: (doc: DocumentListItem) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>已入库文档（{documents.length}）</CardTitle>
      </CardHeader>
      <CardContent>
        {documents.length === 0 ? (
          <p className="text-sm text-zinc-400">暂无文档记录；上传 PDF 后自动出现在此处。</p>
        ) : (
          <div className="max-h-72 space-y-2 overflow-auto">
            {documents.map((doc) => {
              const ingesting = doc.status === "ingesting";
              return (
                <div
                  key={doc.document_id}
                  className={`flex items-start justify-between gap-2 rounded border p-2 text-sm ${
                    ingesting ? "border-amber-200 bg-amber-50" : "border-zinc-200"
                  }`}
                >
                  <div className="min-w-0">
                    <p className="truncate font-medium text-zinc-800">{doc.title || doc.filename}</p>
                    <p className="mt-0.5 text-xs text-zinc-400">
                      id {doc.document_id}
                      {ingesting ? (
                        <> · 入库中…（解析→分块→向量化，完成后自动变为可检索）</>
                      ) : (
                        <>
                          · {doc.page_count} 页 · {doc.node_count} 块
                          {doc.book_id ? ` · ${doc.book_id}` : ""}
                          {doc.chapter_label ? ` · ${doc.chapter_label}` : ""}
                        </>
                      )}
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 shrink-0 border-red-200 px-2 text-xs text-red-600 hover:bg-red-50"
                    disabled={deletingId !== null || ingesting}
                    title={ingesting ? "入库进行中，暂不可删除" : undefined}
                    onClick={() => onDelete(doc)}
                  >
                    {deletingId === doc.document_id ? "删除中…" : "删除"}
                  </Button>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
