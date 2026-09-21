import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

export async function GET(req: NextRequest) {
  const limit = req.nextUrl.searchParams.get("limit") || "500";
  return proxyJson(req, `/agent/api/documents/catalog?limit=${encodeURIComponent(limit)}`);
}
