import { env } from "../env.js";

/**
 * This gateway is the only trusted caller of delivery-gateway's
 * /internal/sign (see apps/delivery-gateway/README.md's trust-boundary
 * note) -- it claims a `viewer` on the caller's behalf, so only code that
 * has already verified the request (requireAuth, below) should call this.
 */
export async function signRenderUrl(artworkId: string, viewer: "anonymous" | "logged_in" | "thumbnail") {
  const res = await fetch(`${env.DELIVERY_GATEWAY_URL}/internal/sign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ artworkId, viewer }),
  });
  if (!res.ok) {
    throw new Error(`delivery-gateway sign failed: ${res.status}`);
  }
  const body = (await res.json()) as { url: string };
  // delivery-gateway returns a path relative to itself, not an absolute
  // URL -- the browser hits delivery-gateway directly (not through this
  // gateway) for the actual image bytes, so it needs the full origin. This
  // is intentionally DELIVERY_GATEWAY_PUBLIC_URL, not the URL just used
  // above to reach delivery-gateway server-to-server -- see env.ts's
  // comment on why those two can differ.
  return `${env.DELIVERY_GATEWAY_PUBLIC_URL ?? env.DELIVERY_GATEWAY_URL}${body.url}`;
}
