import { describe, it, expect, vi } from "vitest";
import { withRetry, isRetryableError } from "../src/lib/retry.js";

describe("isRetryableError", () => {
  it("treats a fetch()-thrown TypeError as retryable", () => {
    expect(isRetryableError(new TypeError("fetch failed"))).toBe(true);
  });

  it("treats a Node system error code (ECONNREFUSED etc) as retryable", () => {
    const err = Object.assign(new Error("connect ECONNREFUSED"), { code: "ECONNREFUSED" });
    expect(isRetryableError(err)).toBe(true);
  });

  it("treats an embedded 5xx status as retryable", () => {
    expect(isRetryableError(new Error("blockchain-svc POST /assets/register failed: 503 Service Unavailable"))).toBe(
      true,
    );
  });

  it("does not retry an embedded 4xx status -- the request itself was wrong", () => {
    expect(isRetryableError(new Error("blockchain-svc POST /assets/register failed: 400 Bad Request"))).toBe(false);
  });

  it("does not retry a plain business-logic error", () => {
    expect(isRetryableError(new Error("content hash collision"))).toBe(false);
  });
});

describe("withRetry", () => {
  it("returns the result on the first success without retrying", async () => {
    const fn = vi.fn().mockResolvedValue("ok");
    const result = await withRetry(fn, { baseDelayMs: 1 });
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("retries a transient failure and eventually succeeds", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockResolvedValue("ok");
    const result = await withRetry(fn, { baseDelayMs: 1, maxDelayMs: 2 });
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("gives up after maxAttempts and throws the last error", async () => {
    const fn = vi.fn().mockRejectedValue(new TypeError("fetch failed"));
    await expect(withRetry(fn, { maxAttempts: 3, baseDelayMs: 1, maxDelayMs: 2 })).rejects.toThrow("fetch failed");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("does not retry a non-transient error -- fails on the first attempt", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("content hash collision"));
    await expect(withRetry(fn, { baseDelayMs: 1 })).rejects.toThrow("content hash collision");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("respects a custom isRetryable predicate", async () => {
    const fn = vi.fn().mockRejectedValueOnce(new Error("retry me")).mockResolvedValue("ok");
    const result = await withRetry(fn, {
      baseDelayMs: 1,
      isRetryable: (err) => err instanceof Error && err.message === "retry me",
    });
    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(2);
  });
});
