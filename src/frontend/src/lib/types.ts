export type ReportLocale = "en" | "zh" | "auto";
export type ResolvedLocale = "en" | "zh";

export interface NarrativeCard {
  card_id?: string | number;
  title?: string;
  body?: string;
  document_id?: number;
  node_id?: string;
  relevance_score?: number;
  relevance_level?: string;
  accn?: string;
}

export interface AskResponse {
  question?: string;
  answer?: string;
  confidence?: number;
  sources_used?: number;
  citation_count?: number;
  limitations?: string | null;
  trace_id?: string | null;
  latency_ms?: number | null;
  pipeline_trace?: Record<string, unknown> | null;
  vertical_scenario?: Record<string, unknown> | null;
  external_evaluation?: {
    filings_observed?: string[];
    [key: string]: unknown;
  } | null;
  evidence_ui?: {
    evidence?: {
      narrative_cards?: NarrativeCard[];
      summary?: Record<string, unknown>;
      [key: string]: unknown;
    };
    conclusion?: Record<string, unknown>;
    risk_panel?: Record<string, unknown>;
    [key: string]: unknown;
  } | null;
  report_locale?: "en" | "zh" | null;
}

export interface HistoryItem {
  id: string;
  traceId?: string | null;
  createdAt: number;
  locale: ResolvedLocale;
  question: string;
  result: AskResponse;
}

export interface PersistedReportSummary {
  trace_id?: string;
  created_at?: string;
  report_locale?: string;
  question?: unknown;
  answer_preview?: unknown;
  [key: string]: unknown;
}

export interface DocumentCatalogItem {
  document_id: number;
  display_name?: string;
  subtitle?: string;
  raw_filename?: string;
  file_type?: string | null;
  node_count?: number;
  raw_title?: string | null;
  source_uri?: string | null;
}
