"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Evidence } from "./types";

/* ── evidence card: text / image + selection checkbox ───────────────── */

export function EvidenceCardView({
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
        /* 同源相对路径，经 next.config.mjs rewrites(/agent/*) 回源后端；
           防穿越由后端 asset store 内聚 */
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
