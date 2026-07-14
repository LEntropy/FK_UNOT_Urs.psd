import { describe, expect, it } from "vitest";
import request from "supertest";
import { createApp } from "../src/app.js";
import { createTestDb } from "./testDb.js";

function freshApp() {
  return createApp(createTestDb());
}

describe("POST /auth/signup", () => {
  it("creates a user with a provisioned custodial wallet and returns tokens", async () => {
    const app = freshApp();
    const res = await request(app)
      .post("/auth/signup")
      .send({ email: "a@example.com", password: "hunter22", handle: "artist_a" });

    expect(res.status).toBe(201);
    expect(res.body.accessToken).toBeTruthy();
    expect(res.body.refreshToken).toBeTruthy();
    expect(res.body.user.email).toBe("a@example.com");
    expect(res.body.user.walletAddress).toMatch(/^0x[0-9a-fA-F]{40}$/);
  });

  it("rejects a duplicate email with 409", async () => {
    const app = freshApp();
    await request(app).post("/auth/signup").send({ email: "dup@example.com", password: "hunter22", handle: "dup1" });
    const res = await request(app)
      .post("/auth/signup")
      .send({ email: "dup@example.com", password: "hunter22", handle: "dup2" });

    expect(res.status).toBe(409);
  });

  it("rejects a short password with 400", async () => {
    const app = freshApp();
    const res = await request(app).post("/auth/signup").send({ email: "b@example.com", password: "short", handle: "b" });
    expect(res.status).toBe(400);
  });
});

describe("POST /auth/login + /me", () => {
  it("logs in with correct credentials and fetches the profile via the access token", async () => {
    const app = freshApp();
    await request(app).post("/auth/signup").send({ email: "c@example.com", password: "hunter22", handle: "creator_c" });

    const login = await request(app).post("/auth/login").send({ email: "c@example.com", password: "hunter22" });
    expect(login.status).toBe(200);

    const me = await request(app).get("/me").set("Authorization", `Bearer ${login.body.accessToken}`);
    expect(me.status).toBe(200);
    expect(me.body.email).toBe("c@example.com");
    expect(me.body.passwordHash).toBeUndefined();
    expect(me.body.encryptedWalletKey).toBeUndefined();
  });

  it("rejects a wrong password with 401", async () => {
    const app = freshApp();
    await request(app).post("/auth/signup").send({ email: "d@example.com", password: "hunter22", handle: "creator_d" });

    const res = await request(app).post("/auth/login").send({ email: "d@example.com", password: "wrong-password" });
    expect(res.status).toBe(401);
  });

  it("rejects /me without a bearer token", async () => {
    const app = freshApp();
    const res = await request(app).get("/me");
    expect(res.status).toBe(401);
  });
});

describe("POST /auth/refresh", () => {
  it("exchanges a valid refresh token for a new access token", async () => {
    const app = freshApp();
    const signup = await request(app)
      .post("/auth/signup")
      .send({ email: "e@example.com", password: "hunter22", handle: "creator_e" });

    const res = await request(app).post("/auth/refresh").send({ refreshToken: signup.body.refreshToken });
    expect(res.status).toBe(200);
    expect(res.body.accessToken).toBeTruthy();
  });

  it("rejects a garbage refresh token", async () => {
    const app = freshApp();
    const res = await request(app).post("/auth/refresh").send({ refreshToken: "not-a-real-token" });
    expect(res.status).toBe(401);
  });
});
