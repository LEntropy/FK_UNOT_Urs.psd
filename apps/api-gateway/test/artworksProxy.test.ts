import { beforeEach, describe, expect, it, vi } from "vitest";
import request from "supertest";
import { createApp } from "../src/app.js";
import { createTestDb } from "./testDb.js";

vi.mock("../src/clients/assetService.js", () => ({
  createArtwork: vi.fn(),
  listArtworks: vi.fn(),
  getArtwork: vi.fn(),
  AssetServiceError: class AssetServiceError extends Error {
    constructor(public status: number, public body: unknown) {
      super("asset-service request failed");
    }
  },
}));

const { createArtwork, listArtworks, getArtwork, AssetServiceError } = await import("../src/clients/assetService.js");

beforeEach(() => {
  vi.clearAllMocks();
});

async function signupAndGetToken(app: ReturnType<typeof createApp>) {
  const res = await request(app)
    .post("/auth/signup")
    .send({ email: "proxy@example.com", password: "hunter22", handle: "proxy_user" });
  return { accessToken: res.body.accessToken as string, walletAddress: res.body.user.walletAddress as string, userId: res.body.user.id as string };
}

describe("artworks proxy", () => {
  it("rejects unauthenticated requests", async () => {
    const app = createApp(createTestDb());
    const res = await request(app).post("/artworks").send({ title: "x", sourceImageUri: "y" });
    expect(res.status).toBe(401);
    expect(createArtwork).not.toHaveBeenCalled();
  });

  it("injects creatorId and ownerWalletAddress from the JWT, never trusting the request body", async () => {
    const app = createApp(createTestDb());
    const { accessToken, walletAddress, userId } = await signupAndGetToken(app);

    vi.mocked(createArtwork).mockResolvedValue({ id: "ast_1", status: "UPLOADED" });

    const res = await request(app)
      .post("/artworks")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({ title: "My Art", sourceImageUri: "/tmp/a.png", creatorId: "someone-else" });

    expect(res.status).toBe(202);
    expect(createArtwork).toHaveBeenCalledWith(
      expect.objectContaining({ title: "My Art", sourceImageUri: "/tmp/a.png" }),
      userId,
      walletAddress,
    );
  });

  it("forwards asset-service's list, scoped to the caller", async () => {
    const app = createApp(createTestDb());
    const { accessToken, userId } = await signupAndGetToken(app);

    vi.mocked(listArtworks).mockResolvedValue([{ id: "ast_1" }]);

    const res = await request(app).get("/artworks").set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(200);
    expect(listArtworks).toHaveBeenCalledWith(userId);
    expect(res.body).toEqual([{ id: "ast_1" }]);
  });

  it("forwards asset-service errors with the same status code", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);

    vi.mocked(getArtwork).mockRejectedValue(new AssetServiceError(404, { error: "no artwork ast_nope" }));

    const res = await request(app).get("/artworks/ast_nope").set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(404);
    expect(res.body).toEqual({ error: "no artwork ast_nope" });
  });
});
