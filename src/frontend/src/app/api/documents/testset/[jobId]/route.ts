import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

/* 轮询评测集生成 / 评测任务状态：{status, questions_count?, error?, questions?, testset_path?, report?} */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  return proxyJson(req, `/agent/api/documents/testset/${encodeURIComponent(jobId)}`);
}
