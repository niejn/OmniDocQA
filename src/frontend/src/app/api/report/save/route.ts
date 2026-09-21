import { NextRequest } from "next/server";
import { proxyJson } from "@/lib/backendProxy";

export async function POST(req: NextRequest) {
  // text() 口径（与 delete 一致）：非法 JSON 不在 Next 层炸成纯文本 500，
  // 而是原样透传给后端拿结构化的 422/400 错误体。
  return proxyJson(req, "/agent/api/report/save", { body: await req.text() });
}
