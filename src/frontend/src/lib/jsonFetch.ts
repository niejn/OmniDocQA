/* 统一的浏览器端 fetch helper（首页 src/lib/api.ts 与 /documents 页共用）。
   - JSON 请求体自动补 Content-Type；FormData 不动 headers，让 fetch 自己生成
     multipart boundary。
   - 非 2xx 抛错，message 优先取后端 `detail`（兼容字符串、{message}/{msg}、
     FastAPI 422 校验数组）。 */

export async function jsonFetch<T>(input: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> =
    init?.body instanceof FormData ? {} : { "Content-Type": "application/json" };
  const res = await fetch(input, { ...init, headers });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    /* non-JSON error body */
  }
  if (!res.ok) {
    throw new Error(extractDetailMessage(data) || `请求失败 (${res.status})`);
  }
  return data as T;
}

/* detail 兼容层：string | {message|msg} | FastAPI 422 校验数组 */
function extractDetailMessage(data: unknown): string | null {
  const detail: unknown = (data as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    const obj = detail as { message?: unknown; msg?: unknown };
    if (typeof obj.message === "string") return obj.message;
    if (typeof obj.msg === "string") return obj.msg;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const msg = (item as { msg?: unknown } | null)?.msg;
        return typeof msg === "string" ? msg : String(item);
      })
      .join("; ");
  }
  return null;
}

/* catch (e) 中 e 为 unknown 的统一取 message 口径，替代散落的
   `e instanceof Error ? e.message : String(e)`。 */
export function toErrorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
