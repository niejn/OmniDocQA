import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ setId: string }> }
) {
  const { setId } = await params;
  return proxyJson(req, `/agent/api/documents/sets/${encodeURIComponent(setId)}`);
}
