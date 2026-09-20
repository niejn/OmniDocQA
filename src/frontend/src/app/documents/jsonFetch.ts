/* Shared fetch helper for the /documents page family.
   - JSON bodies get the JSON Content-Type; FormData is left untouched so fetch
     adds the multipart boundary itself.
   - Non-2xx responses throw with the backend `detail` message when present
     (string, {message}, or FastAPI-style validation arrays). */

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
    const detail: unknown = (data as { detail?: unknown } | null)?.detail;
    let message: string | null = null;
    if (typeof detail === "string") {
      message = detail;
    } else if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      const obj = detail as { message?: unknown; msg?: unknown };
      if (typeof obj.message === "string") message = obj.message;
      else if (typeof obj.msg === "string") message = obj.msg;
    } else if (Array.isArray(detail)) {
      message = detail
        .map((item) => {
          const msg = (item as { msg?: unknown } | null)?.msg;
          return typeof msg === "string" ? msg : String(item);
        })
        .join("; ");
    }
    throw new Error(message || `请求失败 (${res.status})`);
  }
  return data as T;
}
