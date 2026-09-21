import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ documentId: string }> }
) {
  const { documentId } = await params;
  const search = req.nextUrl.searchParams.toString();
  return proxyJson(
    req,
    `/agent/api/documents/chapters/${encodeURIComponent(documentId)}/chunks${search ? `?${search}` : ""}`
  );
}
