import { NextRequest, NextResponse } from "next/server";
import { BACKEND_BASE_URL } from "@/lib/backendProxy";

/* multipart 代理特例：FormData 转发需由 fetch 自动生成带 boundary 的 Content-Type，不走 proxyJson */
export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const upstream = await fetch(`${BACKEND_BASE_URL}/agent/api/documents/upload`, {
      method: "POST",
      body: form,
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
