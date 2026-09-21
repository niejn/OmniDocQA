import { jsonFetch } from "./jsonFetch";
import type {
  AskResponse,
  DocumentCatalogItem,
  PersistedReportSummary,
  ReportLocale,
} from "./types";

function itemsOf<T>(data: unknown): T[] {
  if (Array.isArray(data)) return data;
  if (data !== null && typeof data === "object" && "items" in data) {
    const candidate = (data as Record<string, unknown>)["items"];
    if (Array.isArray(candidate)) return candidate as T[];
  }
  return [];
}

export interface AskGenerateRequest {
  question: string;
  document_ids: number[];
  top_k: number;
  detail_level: "brief" | "detailed" | "comprehensive";
  report_locale: ReportLocale;
  include_pipeline_trace: boolean;
}

export async function askGenerate(payload: AskGenerateRequest): Promise<AskResponse> {
  return jsonFetch<AskResponse>("/api/ask/generate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export interface DocumentGroupsResponse {
  groups?: Record<string, number[]>;
  missing?: boolean;
  path?: string;
}

export async function fetchDocumentCatalog(limit = 500): Promise<DocumentCatalogItem[]> {
  const data = await jsonFetch<unknown>(
    `/api/documents/catalog?limit=${encodeURIComponent(String(limit))}`,
  );
  return itemsOf<DocumentCatalogItem>(data);
}

export async function fetchDocumentGroups(): Promise<DocumentGroupsResponse> {
  return jsonFetch<DocumentGroupsResponse>("/api/documents/groups");
}

export interface ReportDetailResponse {
  response?: AskResponse | null;
  [key: string]: unknown;
}

export async function fetchReportDetail(traceId: string): Promise<ReportDetailResponse> {
  return jsonFetch<ReportDetailResponse>(
    `/api/report/${encodeURIComponent(traceId)}`,
  );
}

export async function fetchReportList(limit = 50): Promise<PersistedReportSummary[]> {
  const data = await jsonFetch<unknown>(
    `/api/report/list?limit=${encodeURIComponent(String(limit))}`,
  );
  return itemsOf<PersistedReportSummary>(data);
}

export interface ReportSavePayload {
  request: AskGenerateRequest;
  response: AskResponse;
  source: string;
}

export async function saveAskReport(payload: ReportSavePayload): Promise<void> {
  await jsonFetch<unknown>("/api/report/save", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function deleteReports(traceIds: string[]): Promise<void> {
  await jsonFetch<unknown>("/api/report/delete", {
    method: "POST",
    body: JSON.stringify({ trace_ids: traceIds }),
  });
}
