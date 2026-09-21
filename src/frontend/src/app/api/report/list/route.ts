import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

export async function GET(req: NextRequest) {
  const limit = req.nextUrl.searchParams.get("limit") || "50";
  return proxyJson(req, `/agent/api/report/list?limit=${encodeURIComponent(limit)}`);
}
