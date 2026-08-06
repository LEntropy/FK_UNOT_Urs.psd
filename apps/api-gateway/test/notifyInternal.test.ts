import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import request from "supertest";
import { createApp } from "../src/app.js";
import { createTestDb } from "./testDb.js";

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

async function signupAndGetUser(app: ReturnType<typeof createApp>) {
  const res = await request(app)
    .post("/auth/signup")
    .send({ email: "creator@example.com", password: "hunter22", handle: "notify_creator" });
  return res.body.user.id as string;
}

describe("POST /internal/notify-evidence-ready", () => {
  it("400s on an invalid body", async () => {
    const app = createApp(createTestDb());
    const res = await request(app).post("/internal/notify-evidence-ready").send({ creatorId: "u1" });
    expect(res.status).toBe(400);
  });

  it("returns {sent: false} for an unknown creatorId, without erroring", async () => {
    const app = createApp(createTestDb());
    const res = await request(app).post("/internal/notify-evidence-ready").send({
      creatorId: "user_does_not_exist",
      artworkTitle: "Starry Fields",
      caseId: "case_abc",
      evidenceType: "copy",
    });
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ sent: false });
  });

  it("returns {sent: false} when SMTP isn't configured, even for a real user", async () => {
    delete process.env.SMTP_HOST;
    delete process.env.SMTP_USER;
    delete process.env.SMTP_PASSWORD;
    const notify = await import("../src/notify.js");
    notify._resetTransportCacheForTests();

    const app = createApp(createTestDb());
    const creatorId = await signupAndGetUser(app);

    const res = await request(app).post("/internal/notify-evidence-ready").send({
      creatorId,
      artworkTitle: "Starry Fields",
      caseId: "case_abc",
      evidenceType: "copy",
    });
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ sent: false });
  });

  it("looks up the creator's real email and sends when SMTP is configured", async () => {
    process.env.SMTP_HOST = "smtp.example.com";
    process.env.SMTP_USER = "user";
    process.env.SMTP_PASSWORD = "pass";
    const fakeSendMail = vi.fn().mockResolvedValue({ messageId: "fake" });
    fakeCreateTransport.mockReturnValue({ sendMail: fakeSendMail });
    const notify = await import("../src/notify.js");
    notify._resetTransportCacheForTests();

    const app = createApp(createTestDb());
    const creatorId = await signupAndGetUser(app);

    const res = await request(app).post("/internal/notify-evidence-ready").send({
      creatorId,
      artworkTitle: "Starry Fields",
      caseId: "case_abc",
      evidenceType: "model_leak",
    });
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ sent: true });
    expect(fakeSendMail).toHaveBeenCalledTimes(1);
    expect(fakeSendMail.mock.calls[0][0].to).toBe("creator@example.com");
  });
});
