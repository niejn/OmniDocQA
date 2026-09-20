import { NextRequest, NextResponse } from "next/server";

const BACKEND_BASE_URL = process.env.BACKEND_API_BASE_URL || "http://127.0.0.1:8000";

/* 轮询评测集生成 / 评测任务状态：{status, questions_count?, error?, questions?, testset_path?, report?} */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  try {
    const { jobId } = await params;
    const upstream = await fetch(`${BACKEND_BASE_URL}/agent/api/documents/testset/${encodeURIComponent(jobId)}`, {
      method: "GET",
      cache: "no-store"
    });
    const text = await upstream.text();
    return new NextResponse(text, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("Content-Type") || "application/json; charset=utf-8" }
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Proxy request failed";
    return NextResponse.json({ detail: { message } }, { status: 500 });
  }
}
