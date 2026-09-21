import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

export async function POST(req: NextRequest) {
  // 保留原实现的 json() 往返：非法 JSON 请求体在此处即 500，不透传给后端
  return proxyJson(req, "/agent/api/report/save", { body: JSON.stringify(await req.json()) });
}
