import { NextRequest, NextResponse } from "next/server";

/*
 * 后端 FastAPI 源。服务端 API 路由（src/app/api/**）共用，运行时读 process.env；
 * 注意 next.config.mjs 的 rewrites 虽然读同一变量，但在 `next build` 时求值并
 * 固化——部署时构建命令必须带 BACKEND_API_BASE_URL（详见 next.config.mjs 注释）。
 */
export const BACKEND_BASE_URL = process.env.BACKEND_API_BASE_URL || "http://127.0.0.1:8000";

interface ProxyJsonInit {
  /** 默认透传 req.method；显式传参仅用于可读性。 */
  method?: string;
  /** 请求体（原始文本）。传入时附带 Content-Type: application/json（与旧路由一致）。 */
  body?: string;
}

/**
 * JSON 代理公共 helper：把 Next API 路由请求转发到后端 `${BACKEND_BASE_URL}{upstreamPath}`。
 *
 * 行为契约（与被替换的各内联实现一致）：
 * - 透传上游 status 与 Content-Type（上游缺失时兜底 application/json; charset=utf-8）；
 * - 响应体原样透传（不解析/不改写）；
 * - 网络/上游不可达异常：500 + 统一错误体 {detail:{message}}。
 *   注意：catalog/groups/ids/report/* 旧实现是 {detail: message}（字符串），本 helper
 *   统一为 {detail:{message}}；客户端 jsonFetch（src/lib/jsonFetch.ts）对两种形状都兼容。
 */
export async function proxyJson(
  req: NextRequest,
  upstreamPath: string,
  init?: ProxyJsonInit
): Promise<NextResponse> {
  try {
    const upstream = await fetch(`${BACKEND_BASE_URL}${upstreamPath}`, {
      method: init?.method ?? req.method,
      ...(init?.body !== undefined
        ? { body: init.body, headers: { "Content-Type": "application/json" } }
        : {}),
      cache: "no-store"
    });
    const text = await upstream.text();
    return new NextResponse(text, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("Content-Type") || "application/json; charset=utf-8"
      }
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Proxy request failed";
    return NextResponse.json({ detail: { message } }, { status: 500 });
  }
}
