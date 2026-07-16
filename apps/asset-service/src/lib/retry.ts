const RETRYABLE_NODE_ERROR_CODES = new Set([
  "ECONNREFUSED",
  "ECONNRESET",
  "ETIMEDOUT",
  "EPIPE",
  "EAI_AGAIN",
  "ENOTFOUND",
]);

/**
 * PROJECT_DESIGN.md §8's "재시도 정책: 아직 미구현" gap -- covers both
 * fetch()-based clients (protectionSvc.ts, blockchainSvc.ts, which throw a
 * TypeError when the network call itself fails, or a plain Error whose
 * message embeds a 5xx status once a response did come back) and
 * infra/kms-adapter's raw TLS socket (imageEncryption.ts's decryptToTempFile),
 * which rejects with the underlying Node system error (ECONNREFUSED etc)
 * rather than a TypeError.
 *
 * Deliberately does NOT retry: 4xx responses (the request itself was wrong;
 * retrying it verbatim fails the same way again), AlreadyRegisteredError
 * (a definitive 409, not a transient fault -- runRegistrationStep already
 * has its own idempotent handling for it), or a protection-svc job that
 * completed with status "failed" (that's a normal return value, not a
 * thrown error -- re-running the same job wastes GPU/CPU time it already
 * legitimately spent).
 */
export function isRetryableError(err: unknown): boolean {
  if (err instanceof TypeError) return true;
  if (err && typeof err === "object" && "code" in err && typeof err.code === "string") {
    if (RETRYABLE_NODE_ERROR_CODES.has(err.code)) return true;
  }
  if (err instanceof Error) {
    const match = err.message.match(/failed: (\d{3})/);
    if (match) return Number(match[1]) >= 500;
  }
  return false;
}

export interface RetryOptions {
  maxAttempts?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  isRetryable?: (err: unknown) => boolean;
}

/**
 * Exponential backoff with full jitter (AWS's "Exponential Backoff and
 * Jitter" post) -- picks a random delay in [0, cap] rather than a fixed
 * cap, so N artworks that all started failing at the same moment (e.g. a
 * blockchain-svc restart) don't all retry in lockstep and re-hammer it the
 * instant it comes back.
 */
export async function withRetry<T>(fn: () => Promise<T>, opts: RetryOptions = {}): Promise<T> {
  const { maxAttempts = 5, baseDelayMs = 1000, maxDelayMs = 30_000, isRetryable = isRetryableError } = opts;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt === maxAttempts || !isRetryable(err)) throw err;
      const cap = Math.min(maxDelayMs, baseDelayMs * 2 ** (attempt - 1));
      await new Promise((resolve) => setTimeout(resolve, Math.random() * cap));
    }
  }
  throw new Error("unreachable");
}
