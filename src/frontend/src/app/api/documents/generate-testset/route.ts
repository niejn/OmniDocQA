import { NextRequest, NextResponse } from "next/server";

const BACKEND_BASE_URL = process.env.BACKEND_API_BASE_URL || "http://127.0.0.1:8000";

/* 生成评测集：{collection?, testset_size} → 202 {job_id}；状态经 GET /api/documents/testset/{job_id} 轮询 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.text();
    const upstream = await fetch(`${BACKEND_BASE_URL}/agent/api/documents/generate-testset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
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
