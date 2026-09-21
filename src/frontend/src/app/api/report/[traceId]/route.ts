import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

export async function GET(req: NextRequest, context: { params: Promise<{ traceId: string }> }) {
  const { traceId } = await context.params;
  const safeTraceId = encodeURIComponent(traceId || "");
  return proxyJson(req, `/agent/api/report/${safeTraceId}`);
}
