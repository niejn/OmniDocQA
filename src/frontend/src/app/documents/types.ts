/* Shared types for the /documents page and its child components. */

/**
 * 检索/入库库 id：内置固定库（text/multimodal）之外，还支持后端动态创建的库
 * （collections 响应中 kind="dynamic" 的条目）。
 */
export type Collection = string;

/* ── 库域规则（"text" 纯文本域 vs 多模态域）───────────────────────────── */

/** 内置纯文本（SEC filings）库 id。 */
export const TEXT_COLLECTION_ID = "text";

/** 该库是否支持筛选器/集合下推（多模态域能力）；text 域后端不接受这些参数。 */
export function supportsFiltersAndSets(id: string): boolean {
  return id !== TEXT_COLLECTION_ID;
}

/* ── collections contract (backend: GET/POST /agent/api/documents/collections) ── */

export type CollectionKind = "fixed" | "dynamic";

export interface CollectionInfo {
  id: string;
  available: boolean;
  points: number | null;
  /** 旧版后端响应可能缺失 kind；消费方统一用 collectionKind() 按 "fixed" 兜底。 */
  kind?: CollectionKind;
}

export interface CollectionsResponse {
  collections: CollectionInfo[];
}

/** 旧响应无 kind 时按 "fixed" 兜底。 */
export function collectionKind(c: CollectionInfo): CollectionKind {
  return c.kind ?? "fixed";
}

/** POST /api/documents/collections 请求体；name 由后端校验 ^[a-z][a-z0-9_]{2,31}$。 */
export interface CreateCollectionPayload {
  name: string;
  description?: string;
}

/* ── testset generation + evaluation contract (202 job + polling) ── */

export type TestsetJobState = "running" | "completed" | "failed";

/** questions 条目：后端可能返回纯文本或带字段的对象，两者都兼容。 */
export type TestsetQuestion =
  | string
  | { question?: string; text?: string; question_text?: string };

/**
 * 评测报告 summary 的一个小节（overall 或按 qtype 分组），形如：
 * { n, errors, hit_rate, gold_recall, mrr, scope_violation?, p95_ms? }。
 * 后端可能演进字段，保留索引签名兜底。
 */
export interface EvalSummarySection {
  n?: number;
  errors?: number;
  hit_rate?: number;
  gold_recall?: number;
  mrr?: number;
  scope_violation?: number;
  p95_ms?: number;
  [key: string]: unknown;
}

export interface EvalReport {
  /**
   * 指标摘要：嵌套结构，顶层键为 "overall" 或 qtype 分组名（如 narrative/sql），
   * 值为 EvalSummarySection。渲染端按小节展开为指标行，未知结构走 raw 兜底。
   */
  summary?: Record<string, EvalSummarySection | unknown>;
  [key: string]: unknown;
}

/** GET /api/documents/testset/{job_id}：生成与评测共用的任务状态响应。 */
export interface TestsetJobStatus {
  status: TestsetJobState;
  job_id?: string;
  questions_count?: number;
  error?: string;
  questions?: TestsetQuestion[];
  /** 生成完成时后端若返回，则前端用它作为运行评测的入参。 */
  testset_path?: string;
  /** 评测完成时的指标报告。 */
  report?: EvalReport | null;
}

/** 202 响应：POST generate-testset / evaluate。 */
export interface StartJobResponse {
  job_id: string;
}

export interface FilterFacetChapter {
  document_id: number;
  chapter_index: number;
  chapter_label: string;
  filename: string;
  pages: number;
  chunks: { text: number; image: number };
}
export interface FilterFacetBook {
  book_id: string;
  chapter_count: number;
  chunk_count: number;
  chapters: FilterFacetChapter[];
}
export interface FiltersFacet {
  books: FilterFacetBook[];
  kinds: string[];
}
export interface DocumentFilters {
  books?: string[];
  chapters?: number[];
  kinds?: ("text" | "image")[];
}
export interface DocumentSet {
  set_id: string;
  name: string;
  kind: "filter" | "enumerated";
  filter_json?: DocumentFilters | null;
  chunk_ids?: string[] | null;
  chunk_count?: number;
  stale_chunk_count?: number;
}
export interface Evidence {
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
export interface AskResult {
  trace_id: string;
  latency_ms?: number;
  collection: Collection;
  question: string;
  answer: string | null;
  evidence: Evidence[];
  counts?: Record<string, number>;
}
export interface ChapterChunks {
  document_id: number;
  kind: string | null;
  page: number;
  page_size: number;
  total: number;
  chunks: Evidence[];
}

/* ── upload contract (backend: POST /agent/api/documents/upload, multipart {file}) ── */

export interface UploadResult {
  document_id: number;
  status: string;
  node_count: number;
  page_count: number;
  filename: string;
}

/* ── ingested-document list contract (backend: GET /agent/api/documents/documents) ── */

export interface DocumentListItem {
  document_id: number;
  filename: string;
  title: string;
  page_count: number;
  node_count: number;
  book_id: string;
  chapter_label: string;
  created_at: string;
}

/* Client-side guard mirroring the backend limit (200MB, .pdf only). */
export const MAX_UPLOAD_BYTES = 200 * 1024 * 1024;

/** 上传目标集合下拉选项。 */
export interface UploadCollectionOption {
  id: string;
  points: number | null;
  /** 后端报告该库可用（Qdrant collection 就绪）。 */
  available: boolean;
  /**
   * true = 当前 uploadCollection 不在后端列表中（新建后列表刷新失败、库被他端删除等），
   * 由前端追加的临时占位选项，展示为「探测中/不可用」，避免 select 空显与静默 422。
   */
  pending?: boolean;
}
