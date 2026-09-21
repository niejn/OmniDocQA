import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

/* 运行评测：{collection?, testset_path} → 202 {job_id}；完成响应含指标报告（report.summary） */
export async function POST(req: NextRequest) {
  return proxyJson(req, "/agent/api/documents/evaluate", { body: await req.text() });
}
