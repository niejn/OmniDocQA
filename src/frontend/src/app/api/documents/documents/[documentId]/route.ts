import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ documentId: string }> }
) {
  const { documentId } = await params;
  return proxyJson(
    req,
    `/agent/api/documents/documents/${encodeURIComponent(documentId)}`
  );
}
