import { beforeEach, describe, expect, it, vi } from "vitest";
import request from "supertest";
import { eq } from "drizzle-orm";
import { createTestDb } from "./testDb.js";

// GOOGLE_CLIENT_ID/SECRET must be set before env.ts (imported transitively
// by app.ts) is loaded -- same reason blockchain-svc's assets.test.ts sets
// process.env before a dynamic import, not a static one at file top.
process.env.GOOGLE_CLIENT_ID = "test-google-client-id";
process.env.GOOGLE_CLIENT_SECRET = "test-google-client-secret";
process.env.WEB_URL = "http://localhost:5173";
process.env.PUBLIC_URL = "http://localhost:4000";
// KAKAO_CLIENT_ID/SECRET deliberately left unset -- covers the
// not-configured path without a second env-juggling test file.

const exchangeCodeForProfile = vi.fn();
vi.mock("../src/auth/oauth.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/auth/oauth.js")>();
  return { ...actual, exchangeCodeForProfile }; // keep the real isProviderConfigured/buildAuthorizeUrl
});

const { createApp } = await import("../src/app.js");
const { users } = await import("../src/db/schema.js");

beforeEach(() => {
  vi.clearAllMocks();
});

describe("GET /auth/:provider", () => {
  it("redirects to Google's authorize URL when configured, with a state param", async () => {
    const app = createApp(createTestDb());
    const res = await request(app).get("/auth/google");
    expect(res.status).toBe(302);
    const location = new URL(res.headers.location);
    expect(location.origin + location.pathname).toBe("https://accounts.google.com/o/oauth2/v2/auth");
    expect(location.searchParams.get("client_id")).toBe("test-google-client-id");
    expect(location.searchParams.get("state")).toBeTruthy();
  });

  it("returns 501 for a provider with no configured credentials", async () => {
    const app = createApp(createTestDb());
    const res = await request(app).get("/auth/kakao");
    expect(res.status).toBe(501);
  });
});

describe("GET /auth/:provider/callback", () => {
  async function getState(app: ReturnType<typeof createApp>): Promise<string> {
    const res = await request(app).get("/auth/google");
    return new URL(res.headers.location).searchParams.get("state")!;
  }

  it("rejects a callback with a missing/invalid state", async () => {
    const app = createApp(createTestDb());
    const res = await request(app).get("/auth/google/callback").query({ code: "abc", state: "not-a-real-state" });
    expect(res.status).toBe(400);
    expect(exchangeCodeForProfile).not.toHaveBeenCalled();
  });

  it("creates a new user and redirects to WEB_URL/oauth-callback with tokens in the hash", async () => {
    const db = createTestDb();
    const app = createApp(db);
    const state = await getState(app);

    exchangeCodeForProfile.mockResolvedValue({
      providerUserId: "google-uid-1",
      email: "newuser@example.com",
      displayName: "New User",
      avatarUri: "https://example.com/pic.jpg",
    });

    const res = await request(app).get("/auth/google/callback").query({ code: "real-code", state });
    expect(res.status).toBe(302);

    const redirect = new URL(res.headers.location);
    expect(redirect.origin + redirect.pathname).toBe("http://localhost:5173/oauth-callback");
    const tokens = new URLSearchParams(redirect.hash.slice(1));
    expect(tokens.get("accessToken")).toBeTruthy();
    expect(tokens.get("refreshToken")).toBeTruthy();

    const row = db.select().from(users).where(eq(users.providerUserId, "google-uid-1")).get()!;
    expect(row.email).toBe("newuser@example.com");
    expect(row.authProvider).toBe("GOOGLE");
    expect(row.passwordHash).toBeNull();
    expect(row.walletAddress).toMatch(/^0x[0-9a-fA-F]{40}$/); // still gets a real custodial wallet
  });

  it("logs in the same user on a second callback instead of creating a duplicate", async () => {
    const db = createTestDb();
    const app = createApp(db);

    exchangeCodeForProfile.mockResolvedValue({
      providerUserId: "google-uid-2",
      email: "repeat@example.com",
      displayName: "Repeat User",
      avatarUri: null,
    });

    const state1 = await getState(app);
    await request(app).get("/auth/google/callback").query({ code: "code1", state: state1 });
    const state2 = await getState(app);
    await request(app).get("/auth/google/callback").query({ code: "code2", state: state2 });

    const rows = db.select().from(users).where(eq(users.providerUserId, "google-uid-2")).all();
    expect(rows).toHaveLength(1);
  });

  it("lets a LOCAL and a GOOGLE account share the same email", async () => {
    const db = createTestDb();
    const app = createApp(db);

    await request(app)
      .post("/auth/signup")
      .send({ email: "shared@example.com", password: "hunter22", handle: "local_shared" });

    exchangeCodeForProfile.mockResolvedValue({
      providerUserId: "google-uid-3",
      email: "shared@example.com",
      displayName: "Shared Email",
      avatarUri: null,
    });
    const state = await getState(app);
    const res = await request(app).get("/auth/google/callback").query({ code: "code", state });

    expect(res.status).toBe(302); // did not collide with the LOCAL account
    const rows = db.select().from(users).where(eq(users.email, "shared@example.com")).all();
    expect(rows).toHaveLength(2);
  });

  it("forwards a provider exchange failure as a 502, not a crash", async () => {
    const db = createTestDb();
    const app = createApp(db);
    const state = await getState(app);

    exchangeCodeForProfile.mockRejectedValue(new Error("Google token exchange failed: 400 invalid_grant"));

    const res = await request(app).get("/auth/google/callback").query({ code: "bad-code", state });
    expect(res.status).toBe(502);
  });
});
