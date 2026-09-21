import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

/* 生成评测集：{collection?, testset_size} → 202 {job_id}；状态经 GET /api/documents/testset/{job_id} 轮询 */
export async function POST(req: NextRequest) {
  return proxyJson(req, "/agent/api/documents/generate-testset", { body: await req.text() });
}
