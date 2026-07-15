import { describe, expect, it, vi } from "vitest";
import request from "supertest";
import { artworks } from "../src/db/schema.js";
import { createTestDb } from "./testDb.js";

vi.mock("../src/orchestration.js", () => ({ runUploadPipeline: vi.fn() }));

const { createApp } = await import("../src/app.js");

function seedArtwork(db: ReturnType<typeof createTestDb>, overrides: Partial<typeof artworks.$inferInsert> = {}) {
  const now = new Date();
  db.insert(artworks)
    .values({
      id: overrides.id ?? "ast_1",
      title: "Test",
      sourceImageUri: "/tmp/a.png",
      creatorId: "creator_a",
      ownerWalletAddress: "0xCD836EEED3Cac282B053c1261f198f9eb848Aab",
      protectionProfile: "L1_PREVIEW",
      allowAiTraining: false,
      watermarkPayloadHex: "deadbeefcafef00d",
      encryptedImagePath: "./data/encrypted/test.enc",
      encryptedDekBase64: "ZmFrZQ==",
      encryptionIv: "ZmFrZQ==",
      encryptionAuthTag: "ZmFrZQ==",
      visibility: "public",
      status: "PUBLISHED",
      publishedAt: now,
      createdAt: now,
      updatedAt: now,
      ...overrides,
    })
    .run();
}

describe("likes", () => {
  it("liking twice is idempotent, not an error", async () => {
    const db = createTestDb();
    seedArtwork(db);
    const app = createApp(db);

    await request(app).post("/artworks/ast_1/likes").send({ userId: "u1" }).expect(204);
    await request(app).post("/artworks/ast_1/likes").send({ userId: "u1" }).expect(204);

    const res = await request(app).get("/artworks/ast_1/likes/count");
    expect(res.body).toEqual({ count: 1 });
  });

  it("unliking removes the like", async () => {
    const db = createTestDb();
    seedArtwork(db);
    const app = createApp(db);

    await request(app).post("/artworks/ast_1/likes").send({ userId: "u1" });
    await request(app).delete("/artworks/ast_1/likes").send({ userId: "u1" }).expect(204);

    const res = await request(app).get("/artworks/ast_1/likes/count");
    expect(res.body).toEqual({ count: 0 });
  });

  it("404s on a nonexistent artwork", async () => {
    const db = createTestDb();
    const app = createApp(db);
    await request(app).post("/artworks/nope/likes").send({ userId: "u1" }).expect(404);
  });
});

describe("bookmarks", () => {
  it("bookmarking into a collection then re-bookmarking updates the collection, not duplicates the row", async () => {
    const db = createTestDb();
    seedArtwork(db);
    const app = createApp(db);

    await request(app).post("/artworks/ast_1/bookmarks").send({ userId: "u1" }).expect(204);
    await request(app).post("/artworks/ast_1/bookmarks").send({ userId: "u1", collectionId: "col_1" }).expect(204);

    const res = await request(app).get("/users/u1/bookmarks");
    expect(res.body).toHaveLength(1);
    expect(res.body[0].collectionId).toBe("col_1");
  });
});

describe("follows", () => {
  it("cannot follow yourself", async () => {
    const db = createTestDb();
    const app = createApp(db);
    await request(app).post("/users/u1/follow").send({ userId: "u1" }).expect(400);
  });

  it("follow then unfollow updates the follower count", async () => {
    const db = createTestDb();
    const app = createApp(db);

    await request(app).post("/users/creator_a/follow").send({ userId: "u1" }).expect(204);
    let res = await request(app).get("/users/creator_a/followers/count");
    expect(res.body).toEqual({ count: 1 });

    await request(app).delete("/users/creator_a/follow").send({ userId: "u1" }).expect(204);
    res = await request(app).get("/users/creator_a/followers/count");
    expect(res.body).toEqual({ count: 0 });
  });
});

describe("comments", () => {
  it("posts and lists comments newest-first", async () => {
    const db = createTestDb();
    seedArtwork(db);
    const app = createApp(db);

    await request(app).post("/artworks/ast_1/comments").send({ userId: "u1", body: "first" }).expect(201);
    await request(app).post("/artworks/ast_1/comments").send({ userId: "u2", body: "second" }).expect(201);

    const res = await request(app).get("/artworks/ast_1/comments");
    expect(res.body.map((c: { body: string }) => c.body)).toEqual(["second", "first"]);
  });
});

describe("reports / moderation", () => {
  it("a report starts PENDING and shows up in the moderation queue", async () => {
    const db = createTestDb();
    seedArtwork(db);
    const app = createApp(db);

    const created = await request(app)
      .post("/artworks/ast_1/reports")
      .send({ reporterId: "u1", reason: "stolen artwork" })
      .expect(201);
    expect(created.body.status).toBe("PENDING");

    const queue = await request(app).get("/moderation/reports");
    expect(queue.body).toHaveLength(1);
    expect(queue.body[0].id).toBe(created.body.id);
  });

  it("resolving a report is one-way -- can't resolve an already-resolved report", async () => {
    const db = createTestDb();
    seedArtwork(db);
    const app = createApp(db);

    const created = await request(app).post("/artworks/ast_1/reports").send({ reporterId: "u1", reason: "spam" });
    await request(app).patch(`/moderation/reports/${created.body.id}`).send({ status: "DISMISSED" }).expect(200);
    await request(app).patch(`/moderation/reports/${created.body.id}`).send({ status: "RESOLVED" }).expect(409);
  });
});

describe("feed", () => {
  it("latest feed only returns public+published artworks, newest first", async () => {
    const db = createTestDb();
    const t1 = new Date(2020, 0, 1);
    const t2 = new Date(2020, 0, 2);
    seedArtwork(db, { id: "ast_old", publishedAt: t1, createdAt: t1, updatedAt: t1 });
    seedArtwork(db, { id: "ast_new", publishedAt: t2, createdAt: t2, updatedAt: t2 });
    seedArtwork(db, { id: "ast_private", visibility: "private" });
    seedArtwork(db, { id: "ast_unpublished", status: "UPLOADED", publishedAt: null });
    const app = createApp(db);

    const res = await request(app).get("/feed?type=latest");
    expect(res.body.map((a: { id: string }) => a.id)).toEqual(["ast_new", "ast_old"]);
  });

  it("following feed requires userId and only includes followed creators", async () => {
    const db = createTestDb();
    seedArtwork(db, { id: "ast_followed", creatorId: "creator_followed" });
    seedArtwork(db, { id: "ast_not_followed", creatorId: "creator_other" });
    const app = createApp(db);

    await request(app).get("/feed?type=following").expect(400);

    await request(app).post("/users/creator_followed/follow").send({ userId: "viewer" });
    const res = await request(app).get("/feed?type=following&userId=viewer");
    expect(res.body.map((a: { id: string }) => a.id)).toEqual(["ast_followed"]);
  });

  it("popular feed orders by like count", async () => {
    const db = createTestDb();
    seedArtwork(db, { id: "ast_liked_twice" });
    seedArtwork(db, { id: "ast_liked_once" });
    const app = createApp(db);

    await request(app).post("/artworks/ast_liked_twice/likes").send({ userId: "u1" });
    await request(app).post("/artworks/ast_liked_twice/likes").send({ userId: "u2" });
    await request(app).post("/artworks/ast_liked_once/likes").send({ userId: "u1" });

    const res = await request(app).get("/feed?type=popular");
    expect(res.body.map((a: { id: string }) => a.id)).toEqual(["ast_liked_twice", "ast_liked_once"]);
  });
});
