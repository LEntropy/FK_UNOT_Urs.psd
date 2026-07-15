const BASE_URL = import.meta.env.VITE_API_GATEWAY_URL ?? "http://localhost:4000";

/** Plain <a> links, not client-side navigation -- these need a real
 * top-level browser redirect to leave the SPA and hit the provider. */
export function OAuthButtons() {
  return (
    <div className="mt-4 flex flex-col gap-2">
      <div className="flex items-center gap-2 text-xs text-neutral-500">
        <div className="h-px flex-1 bg-neutral-800" />
        또는
        <div className="h-px flex-1 bg-neutral-800" />
      </div>
      <a
        href={`${BASE_URL}/auth/google`}
        className="rounded border border-neutral-700 px-3 py-2 text-center font-medium hover:bg-neutral-900"
      >
        Google로 계속하기
      </a>
      <a
        href={`${BASE_URL}/auth/kakao`}
        className="rounded border border-neutral-700 px-3 py-2 text-center font-medium hover:bg-neutral-900"
      >
        카카오로 계속하기
      </a>
    </div>
  );
}
