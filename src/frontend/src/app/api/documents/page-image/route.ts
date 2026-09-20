import { NextRequest, NextResponse } from "next/server";

const BACKEND_BASE_URL = process.env.BACKEND_API_BASE_URL || "http://127.0.0.1:8000";

export async function GET(req: NextRequest) {
  try {
    const documentId = req.nextUrl.searchParams.get("document_id") || "";
    const name = req.nextUrl.searchParams.get("name") || "";
    const upstream = await fetch(
      `${BACKEND_BASE_URL}/agent/api/documents/page-image?document_id=${encodeURIComponent(documentId)}&name=${encodeURIComponent(name)}`,
      { method: "GET", cache: "no-store" }
    );
    if (!upstream.ok) {
      const text = await upstream.text();
      return new NextResponse(text || upstream.statusText, {
        status: upstream.status,
        headers: { "Content-Type": upstream.headers.get("Content-Type") || "application/json; charset=utf-8" }
      });
    }
    const buffer = await upstream.arrayBuffer();
    return new NextResponse(buffer, {
      status: 200,
      headers: {
        "Content-Type": upstream.headers.get("Content-Type") || "image/jpeg",
        "Cache-Control": "public, max-age=3600"
      }
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Proxy request failed";
    return NextResponse.json({ detail: { message } }, { status: 500 });
  }
}
