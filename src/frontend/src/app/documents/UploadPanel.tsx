"use client";

import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { MAX_UPLOAD_BYTES, UploadCollectionOption } from "./types";

/* ── upload panel: drag & drop + file picker + target collection select + inline create ── */

/** 与后端校验一致的集合名规则：^[a-z][a-z0-9_]{2,31}$（前端预检，后端仍为最终校验方）。 */
const COLLECTION_NAME_PATTERN = /^[a-z][a-z0-9_]{2,31}$/;

export function UploadPanel({
  uploading,
  onUpload,
  onError,
  collectionOptions,
  collection,
  onCollectionChange,
  onCreateCollection,
  creatingCollection
}: {
  uploading: boolean;
  onUpload: (file: File) => void;
  onError: (message: string) => void;
  /** 可写集合（multimodal 固定库 + 后端 dynamic 且 available 的库）；pending 项为临时占位。 */
  collectionOptions: UploadCollectionOption[];
  collection: string;
  onCollectionChange: (id: string) => void;
  /** 创建成功返回 true（调用方负责刷新列表并选中新集合）。 */
  onCreateCollection: (name: string, description: string) => Promise<boolean>;
  creatingCollection: boolean;
}) {
  const [dragOver, setDragOver] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);
  // P2-10: 创建按钮同帧双击穿透锁——creatingCollection 状态置位要等下一次渲染，
  // 同步连点两下都能通过 `if (creatingCollection) return`，用 ref 在同步路径先挡住第二击。
  const createLockRef = useRef(false);

  const validateAndSubmit = (file: File | null | undefined) => {
    if (!file || uploading) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      onError(`仅支持 .pdf 文件，收到「${file.name}」`);
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      const mb = (file.size / 1024 / 1024).toFixed(1);
      onError(`文件超过 200MB 上限（当前 ${mb}MB）`);
      return;
    }
    onUpload(file);
  };

  const submitCreate = async () => {
    const name = newName.trim();
    if (!name || creatingCollection) return;
    if (!COLLECTION_NAME_PATTERN.test(name)) {
      onError("集合名不符合规则：小写字母开头，仅含小写字母/数字/下划线，长度 3-32（如 finances_2024）");
      return;
    }
    if (createLockRef.current) return; // P2-10: 同帧双击第二击直接忽略
    createLockRef.current = true;
    try {
      const ok = await onCreateCollection(name, newDesc.trim());
      if (ok) {
        setNewName("");
        setNewDesc("");
        setShowCreate(false);
      }
    } finally {
      createLockRef.current = false; // 失败也要释放锁，允许用户重试
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>上传 PDF 入库</CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className={cn(
            "rounded-md border-2 border-dashed p-4 text-center text-sm transition-colors",
            dragOver ? "border-sky-400 bg-sky-50" : "border-zinc-300 bg-zinc-50",
            uploading && "opacity-60"
          )}
          onDragOver={(e) => {
            e.preventDefault();
            if (!uploading) setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            if (uploading) return;
            validateAndSubmit(e.dataTransfer.files?.[0]);
          }}
        >
          {uploading ? (
            <p className="py-2 text-zinc-500">上传中…（入库完成后会自动刷新列表）</p>
          ) : (
            <>
              <p className="text-zinc-500">拖拽 PDF 到此处，或</p>
              <Button
                className="mt-2"
                variant="outline"
                size="sm"
                disabled={uploading}
                onClick={() => inputRef.current?.click()}
              >
                选择文件
              </Button>
            </>
          )}
          <p className="mt-2 text-xs text-zinc-400">仅 .pdf · ≤200MB（后端另限 500 页；重复 document_id 不替换）</p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          className="hidden"
          onChange={(e) => {
            validateAndSubmit(e.target.files?.[0]);
            e.target.value = ""; // 允许重复选择同一文件
          }}
        />
        {/* 目标集合选择 + 新建集合入口 */}
        <div className="mt-3 space-y-2 border-t border-zinc-100 pt-3 text-sm">
          <div className="flex items-center gap-2">
            <span className="shrink-0 text-zinc-500">入库集合</span>
            <select
              className="min-w-0 flex-1 rounded border border-zinc-300 px-2 py-1 text-sm"
              value={collection}
              disabled={uploading}
              onChange={(e) => onCollectionChange(e.target.value)}
            >
              {collectionOptions.map((o) => (
                <option key={o.id} value={o.id}>
                  {/* P2-8/9: 不可用库与前端追加的临时占位（不在后端列表中）都要有文案标注，
                      避免用户在 502/422 后才发现目标库无效。 */}
                  {o.id}（{o.pending ? "?" : o.points ?? 0} 块）
                  {o.pending ? "（探测中/不可用）" : !o.available ? "（不可用）" : ""}
                </option>
              ))}
            </select>
          </div>
          {showCreate ? (
            <div className="space-y-2">
              <input
                className="w-full rounded border border-zinc-300 px-2 py-1 text-sm"
                placeholder="集合名（小写字母开头，3-32 位）"
                value={newName}
                autoFocus
                onChange={(e) => setNewName(e.target.value)}
              />
              <input
                className="w-full rounded border border-zinc-300 px-2 py-1 text-sm"
                placeholder="描述（可选）"
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
              />
              <div className="flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => setShowCreate(false)}>
                  取消
                </Button>
                <Button
                  size="sm"
                  disabled={creatingCollection || !newName.trim()}
                  onClick={() => void submitCreate()}
                >
                  {creatingCollection ? "创建中…" : "创建"}
                </Button>
              </div>
            </div>
          ) : (
            <button
              className="text-xs text-sky-600 underline"
              disabled={creatingCollection}
              onClick={() => setShowCreate(true)}
            >
              + 新建集合
            </button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
