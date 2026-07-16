import { useAuthStore } from "../store/auth";

const BASE_URL = import.meta.env.VITE_API_GATEWAY_URL ?? "http://localhost:4000";

export class ApiError extends Error {
  constructor(public status: number, public body: unknown) {
    super(`API request failed: ${status}`);
  }
}

async function rawFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const accessToken = useAuthStore.getState().accessToken;
  const headers = new Headers(init.headers);
  // A FormData body must NOT get an explicit Content-Type -- the browser
  // sets its own `multipart/form-data; boundary=...` when it sees the body
  // is a FormData instance, and a caller-set "application/json" here would
  // silently break every multipart upload's boundary parsing server-side.
  if (!(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);

  return fetch(`${BASE_URL}${path}`, { ...init, headers });
}

/** Retries once with a refreshed access token on a 401, then gives up and logs out. */
async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res = await rawFetch(path, init);

  if (res.status === 401 && useAuthStore.getState().refreshToken) {
    const refreshed = await tryRefresh();
    if (refreshed) {
      res = await rawFetch(path, init);
    }
  }

  const body = await res.json().catch(() => undefined);
  if (!res.ok) {
    if (res.status === 401) useAuthStore.getState().logout();
    throw new ApiError(res.status, body);
  }
  return body as T;
}

async function tryRefresh(): Promise<boolean> {
  const refreshToken = useAuthStore.getState().refreshToken;
  if (!refreshToken) return false;

  const res = await fetch(`${BASE_URL}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refreshToken }),
  });
  if (!res.ok) return false;

  const body = await res.json();
  useAuthStore.getState().setAccessToken(body.accessToken);
  return true;
}

export const api = {
  get: <T>(path: string) => apiFetch<T>(path),
  post: <T>(path: string, body?: unknown) =>
    apiFetch<T>(path, { method: "POST", body: body !== undefined ? JSON.stringify(body) : undefined }),
  patch: <T>(path: string, body?: unknown) =>
    apiFetch<T>(path, { method: "PATCH", body: body !== undefined ? JSON.stringify(body) : undefined }),
  delete: <T>(path: string, body?: unknown) =>
    apiFetch<T>(path, { method: "DELETE", body: body !== undefined ? JSON.stringify(body) : undefined }),
  // Real file uploads (UploadPage) -- takes a FormData directly, never
  // JSON.stringify'd (see rawFetch's FormData check above for why that
  // matters: the browser needs to set its own multipart boundary header).
  upload: <T>(path: string, form: FormData) => apiFetch<T>(path, { method: "POST", body: form }),
};
