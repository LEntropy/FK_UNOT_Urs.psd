import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const fakeCreateTransport = vi.fn();

vi.mock("nodemailer", () => ({
  default: { createTransport: (...args: unknown[]) => fakeCreateTransport(...args) },
  createTransport: (...args: unknown[]) => fakeCreateTransport(...args),
}));

const originalEnv = { ...process.env };

beforeEach(() => {
  fakeCreateTransport.mockReset();
});

afterEach(async () => {
  process.env = { ...originalEnv };
  const notify = await import("../src/notify.js");
  notify._resetTransportCacheForTests();
});

describe("sendEvidenceReadyEmail", () => {
  it("returns false without sending anything when SMTP isn't configured", async () => {
    delete process.env.SMTP_HOST;
    delete process.env.SMTP_USER;
    delete process.env.SMTP_PASSWORD;
    const notify = await import("../src/notify.js");
    notify._resetTransportCacheForTests();

    const sent = await notify.sendEvidenceReadyEmail({
      toEmail: "creator@example.com",
      artworkTitle: "Starry Fields",
      caseId: "case_abc",
      evidenceType: "copy",
    });

    expect(sent).toBe(false);
    expect(fakeCreateTransport).not.toHaveBeenCalled();
  });

  it("sends via the configured transport when SMTP env vars are set", async () => {
    process.env.SMTP_HOST = "smtp.example.com";
    process.env.SMTP_USER = "user";
    process.env.SMTP_PASSWORD = "pass";

    const fakeSendMail = vi.fn().mockResolvedValue({ messageId: "fake" });
    fakeCreateTransport.mockReturnValue({ sendMail: fakeSendMail });

    const notify = await import("../src/notify.js");
    notify._resetTransportCacheForTests();

    const sent = await notify.sendEvidenceReadyEmail({
      toEmail: "creator@example.com",
      artworkTitle: "Starry Fields",
      caseId: "case_abc",
      evidenceType: "model_leak",
    });

    expect(sent).toBe(true);
    expect(fakeSendMail).toHaveBeenCalledTimes(1);
    const call = fakeSendMail.mock.calls[0][0];
    expect(call.to).toBe("creator@example.com");
    expect(call.subject).toContain("학습 유출");
    expect(call.subject).toContain("Starry Fields");
  });

  it("returns false (not throw) when the transport's sendMail rejects", async () => {
    process.env.SMTP_HOST = "smtp.example.com";
    process.env.SMTP_USER = "user";
    process.env.SMTP_PASSWORD = "pass";

    const fakeSendMail = vi.fn().mockRejectedValue(new Error("connection refused"));
    fakeCreateTransport.mockReturnValue({ sendMail: fakeSendMail });

    const notify = await import("../src/notify.js");
    notify._resetTransportCacheForTests();

    const sent = await notify.sendEvidenceReadyEmail({
      toEmail: "creator@example.com",
      artworkTitle: "Starry Fields",
      caseId: "case_abc",
      evidenceType: "copy",
    });

    expect(sent).toBe(false);
  });
});
