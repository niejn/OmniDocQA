import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

export async function GET(req: NextRequest) {
  return proxyJson(req, "/agent/api/documents/collections");
}

/* 创建动态集合：{name, description?}；后端校验 ^[a-z][a-z0-9_]{2,31}$、重名 422 */
export async function POST(req: NextRequest) {
  return proxyJson(req, "/agent/api/documents/collections", { body: await req.text() });
}
