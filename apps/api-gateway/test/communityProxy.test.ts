import { beforeEach, describe, expect, it, vi } from "vitest";
import request from "supertest";
import { createApp } from "../src/app.js";
import { createTestDb } from "./testDb.js";

vi.mock("../src/clients/community.js", () => ({
  like: vi.fn(),
  unlike: vi.fn(),
  likeCount: vi.fn(),
  bookmark: vi.fn(),
  unbookmark: vi.fn(),
  listBookmarks: vi.fn(),
  createCollection: vi.fn(),
  listCollections: vi.fn(),
  follow: vi.fn(),
  unfollow: vi.fn(),
  followerCount: vi.fn(),
  createComment: vi.fn(),
  listComments: vi.fn(),
  createReport: vi.fn(),
  listModerationQueue: vi.fn(),
  resolveReport: vi.fn(),
  getFeed: vi.fn(),
}));
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

const community = await import("../src/clients/community.js");
const { AssetServiceError } = await import("../src/clients/assetService.js");

beforeEach(() => {
  vi.clearAllMocks();
});

async function signupAndGetToken(app: ReturnType<typeof createApp>, email = "community@example.com", handle = "community_user") {
  const res = await request(app).post("/auth/signup").send({ email, password: "hunter22", handle });
  return { accessToken: res.body.accessToken as string, userId: res.body.user.id as string };
}

describe("community proxy: identity injection", () => {
  it("rejects unauthenticated requests", async () => {
    const app = createApp(createTestDb());
    await request(app).post("/artworks/ast_1/likes").expect(401);
    expect(community.like).not.toHaveBeenCalled();
  });

  it("likes as the JWT's own user, ignoring any userId in the body", async () => {
    const app = createApp(createTestDb());
    const { accessToken, userId } = await signupAndGetToken(app);
    vi.mocked(community.like).mockResolvedValue(undefined);

    const res = await request(app)
      .post("/artworks/ast_1/likes")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({ userId: "someone-else" });

    expect(res.status).toBe(204);
    expect(community.like).toHaveBeenCalledWith("ast_1", userId);
  });

  it("comments as the JWT's own user", async () => {
    const app = createApp(createTestDb());
    const { accessToken, userId } = await signupAndGetToken(app);
    vi.mocked(community.createComment).mockResolvedValue({ id: "cmt_1" });

    const res = await request(app)
      .post("/artworks/ast_1/comments")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({ body: "nice work" });

    expect(res.status).toBe(201);
    expect(community.createComment).toHaveBeenCalledWith("ast_1", userId, "nice work");
  });

  it("the following feed always uses the caller's own id, never a query param", async () => {
    const app = createApp(createTestDb());
    const { accessToken, userId } = await signupAndGetToken(app);
    vi.mocked(community.getFeed).mockResolvedValue([]);

    await request(app)
      .get("/feed?type=following&userId=someone-else")
      .set("Authorization", `Bearer ${accessToken}`)
      .expect(200);

    expect(community.getFeed).toHaveBeenCalledWith("following", userId, undefined);
  });

  it("forwards asset-service errors with the same status code", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);
    vi.mocked(community.like).mockRejectedValue(new AssetServiceError(404, { error: "no artwork ast_nope" }));

    const res = await request(app)
      .post("/artworks/ast_nope/likes")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({});
    expect(res.status).toBe(404);
    expect(res.body).toEqual({ error: "no artwork ast_nope" });
  });
});

describe("community proxy: moderation is role-gated", () => {
  it("a plain CREATOR cannot see the moderation queue", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);

    const res = await request(app).get("/moderation/reports").set("Authorization", `Bearer ${accessToken}`);
    expect(res.status).toBe(403);
    expect(community.listModerationQueue).not.toHaveBeenCalled();
  });

  it("a plain CREATOR cannot resolve a report", async () => {
    const app = createApp(createTestDb());
    const { accessToken } = await signupAndGetToken(app);

    const res = await request(app)
      .patch("/moderation/reports/rpt_1")
      .set("Authorization", `Bearer ${accessToken}`)
      .send({ status: "RESOLVED" });
    expect(res.status).toBe(403);
    expect(community.resolveReport).not.toHaveBeenCalled();
  });
});
