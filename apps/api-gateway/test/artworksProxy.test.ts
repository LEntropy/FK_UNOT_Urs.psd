import { beforeEach, describe, expect, it, vi } from "vitest";
import request from "supertest";
import { createApp } from "../src/app.js";
import { createTestDb } from "./testDb.js";

vi.mock("../src/clients/assetService.js", () => ({
  createArtwork: vi.fn(),
  createArtworkWithFile: vi.fn(),
  listArtworks: vi.fn(),
  getArtwork: vi.fn(),
  AssetServiceError: class AssetServiceError extends Error {
    constructor(public status: number, public body: unknown) {
      super("asset-service request failed");
    }
  },
}));
vi.mock("../src/clients/deliveryGateway.js", () => ({ signRenderUrl: vi.fn() }));

const { createArtwork, createArtworkWithFile, listArtworks, getArtwork, AssetServiceError } = await import("../src/clients/assetService.js");
const { signRenderUrl } = await import("../src/clients/deliveryGateway.js");

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

  it("routes a real multipart file upload to createArtworkWithFile, not createArtwork", async () => {
    const app = createApp(createTestDb());
    const { accessToken, walletAddress, userId } = await signupAndGetToken(app);

    vi.mocked(createArtworkWithFile).mockResolvedValue({ id: "ast_2", status: "UPLOADED" });

    const res = await request(app)
      .post("/artworks")
      .set("Authorization", `Bearer ${accessToken}`)
      .field("title", "Mona Lisa")
      .field("creatorId", "someone-else") // still must be ignored, same as the JSON path
      .attach("image", Buffer.from("fake image bytes"), "mona_lisa.jpg");

    expect(res.status).toBe(202);
    expect(res.body).toEqual({ id: "ast_2", status: "UPLOADED" });
    expect(createArtwork).not.toHaveBeenCalled();
    expect(createArtworkWithFile).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Mona Lisa",
        file: expect.objectContaining({ originalname: "mona_lisa.jpg" }),
      }),
      userId,
      walletAddress,
    );
  });

  it("400s when a multipart request has neither a file nor sourceImageUri", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);

    const res = await request(app)
      .post("/artworks")
      .set("Authorization", `Bearer ${accessToken}`)
      .field("title", "Nothing attached");

    expect(res.status).toBe(400);
    expect(createArtwork).not.toHaveBeenCalled();
    expect(createArtworkWithFile).not.toHaveBeenCalled();
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

describe("GET /artworks/:id/render-url", () => {
  it("rejects unauthenticated requests", async () => {
    const app = createApp(createTestDb());
    const res = await request(app).get("/artworks/ast_1/render-url");
    expect(res.status).toBe(401);
    expect(signRenderUrl).not.toHaveBeenCalled();
  });

  it("defaults to the logged_in viewer for an authenticated caller", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(signRenderUrl).mockResolvedValue("http://localhost:4500/asset/ast_1/render?variant=public_preview_2048&exp=1&sig=a");

    const res = await request(app).get("/artworks/ast_1/render-url").set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(200);
    expect(signRenderUrl).toHaveBeenCalledWith("ast_1", "logged_in");
    expect(res.body.url).toContain("public_preview_2048");
  });

  it("passes through the thumbnail variant when requested", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(signRenderUrl).mockResolvedValue("http://localhost:4500/asset/ast_1/render?variant=grid_thumbnail_512&exp=1&sig=a");

    const res = await request(app)
      .get("/artworks/ast_1/render-url?variant=thumbnail")
      .set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(200);
    expect(signRenderUrl).toHaveBeenCalledWith("ast_1", "thumbnail");
  });

  it("502s when delivery-gateway is unreachable", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(signRenderUrl).mockRejectedValue(new Error("connect ECONNREFUSED"));

    const res = await request(app).get("/artworks/ast_1/render-url").set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(502);
  });
});
