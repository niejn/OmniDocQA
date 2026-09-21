import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

export async function GET(req: NextRequest) {
  return proxyJson(req, "/agent/api/documents/documents");
}
