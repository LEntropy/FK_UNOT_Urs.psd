import { randomUUID } from "node:crypto";
import { Router } from "express";
import { and, desc, eq, sql } from "drizzle-orm";
import { z } from "zod";
import type { Db } from "../db/client.js";
import { artworks, bookmarks, collections, comments, follows, likes, reports } from "../db/schema.js";

/**
 * PROJECT_DESIGN.md §3-2 community features: feed, follow, like, bookmark,
 * comment, report -> moderation queue. Same trust boundary as
 * routes/artworks.ts -- this service takes userId/creatorId/reporterId as
 * given in the request, no auth of its own. api-gateway is the only place
 * that verifies identity (it injects the real id from the JWT before
 * proxying here); calling this service directly, unproxied, means the
 * caller is trusted to tell the truth about who they are -- fine for a
 * PoC's internal service-to-service boundary, not for a public-facing one.
 */

const userIdBody = z.object({ userId: z.string().min(1) });

export function communityRouter(db: Db): Router {
  const router = Router();

  // --- likes ---------------------------------------------------------
  router.post("/artworks/:id/likes", (req, res) => {
    const parsed = userIdBody.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
    if (!artworkExists(db, req.params.id)) return res.status(404).json({ error: `no artwork ${req.params.id}` });

    // Idempotent: liking twice isn't an error, it's a no-op (INSERT OR
    // IGNORE via the unique index), matching "like" being a toggle-state
    // concept, not an event log.
    db.insert(likes)
      .values({ userId: parsed.data.userId, artworkId: req.params.id, createdAt: new Date() })
      .onConflictDoNothing()
      .run();
    res.status(204).send();
  });

  router.delete("/artworks/:id/likes", (req, res) => {
    const parsed = userIdBody.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

    db.delete(likes)
      .where(and(eq(likes.userId, parsed.data.userId), eq(likes.artworkId, req.params.id)))
      .run();
    res.status(204).send();
  });

  router.get("/artworks/:id/likes/count", (req, res) => {
    const row = db
      .select({ count: sql<number>`count(*)` })
      .from(likes)
      .where(eq(likes.artworkId, req.params.id))
      .get();
    res.json({ count: row?.count ?? 0 });
  });

  // --- bookmarks -------------------------------------------------------
  const bookmarkBody = userIdBody.extend({ collectionId: z.string().min(1).optional() });

  router.post("/artworks/:id/bookmarks", (req, res) => {
    const parsed = bookmarkBody.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
    if (!artworkExists(db, req.params.id)) return res.status(404).json({ error: `no artwork ${req.params.id}` });

    db.insert(bookmarks)
      .values({
        userId: parsed.data.userId,
        artworkId: req.params.id,
        collectionId: parsed.data.collectionId ?? null,
        createdAt: new Date(),
      })
      .onConflictDoUpdate({
        target: [bookmarks.userId, bookmarks.artworkId],
        set: { collectionId: parsed.data.collectionId ?? null },
      })
      .run();
    res.status(204).send();
  });

  router.delete("/artworks/:id/bookmarks", (req, res) => {
    const parsed = userIdBody.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

    db.delete(bookmarks)
      .where(and(eq(bookmarks.userId, parsed.data.userId), eq(bookmarks.artworkId, req.params.id)))
      .run();
    res.status(204).send();
  });

  router.get("/users/:userId/bookmarks", (req, res) => {
    const rows = db.select().from(bookmarks).where(eq(bookmarks.userId, req.params.userId)).all();
    res.json(rows);
  });

  // --- collections -----------------------------------------------------
  const createCollectionSchema = z.object({
    userId: z.string().min(1),
    name: z.string().min(1),
    isPublic: z.boolean().default(true),
  });

  router.post("/collections", (req, res) => {
    const parsed = createCollectionSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

    const id = `col_${randomUUID().replace(/-/g, "").slice(0, 16)}`;
    db.insert(collections)
      .values({ id, userId: parsed.data.userId, name: parsed.data.name, isPublic: parsed.data.isPublic, createdAt: new Date() })
      .run();
    res.status(201).json({ id });
  });

  router.get("/collections", (req, res) => {
    const userId = typeof req.query.userId === "string" ? req.query.userId : undefined;
    const rows = userId
      ? db.select().from(collections).where(eq(collections.userId, userId)).all()
      : db.select().from(collections).where(eq(collections.isPublic, true)).all();
    res.json(rows);
  });

  // --- follows -----------------------------------------------------------
  router.post("/users/:creatorId/follow", (req, res) => {
    const parsed = userIdBody.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
    if (parsed.data.userId === req.params.creatorId) {
      return res.status(400).json({ error: "cannot follow yourself" });
    }

    db.insert(follows)
      .values({ followerId: parsed.data.userId, creatorId: req.params.creatorId, createdAt: new Date() })
      .onConflictDoNothing()
      .run();
    res.status(204).send();
  });

  router.delete("/users/:creatorId/follow", (req, res) => {
    const parsed = userIdBody.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

    db.delete(follows)
      .where(and(eq(follows.followerId, parsed.data.userId), eq(follows.creatorId, req.params.creatorId)))
      .run();
    res.status(204).send();
  });

  router.get("/users/:creatorId/followers/count", (req, res) => {
    const row = db
      .select({ count: sql<number>`count(*)` })
      .from(follows)
      .where(eq(follows.creatorId, req.params.creatorId))
      .get();
    res.json({ count: row?.count ?? 0 });
  });

  // --- comments ----------------------------------------------------------
  const createCommentSchema = z.object({ userId: z.string().min(1), body: z.string().min(1).max(2000) });

  router.post("/artworks/:id/comments", (req, res) => {
    const parsed = createCommentSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
    if (!artworkExists(db, req.params.id)) return res.status(404).json({ error: `no artwork ${req.params.id}` });

    const id = `cmt_${randomUUID().replace(/-/g, "").slice(0, 16)}`;
    db.insert(comments)
      .values({ id, artworkId: req.params.id, userId: parsed.data.userId, body: parsed.data.body, createdAt: new Date() })
      .run();
    res.status(201).json({ id });
  });

  router.get("/artworks/:id/comments", (req, res) => {
    // createdAt alone ties when two comments land in the same millisecond
    // (real under load, not just a test artifact); rowid as a tiebreaker
    // preserves actual insertion order since SQLite assigns it monotonically.
    const rows = db
      .select()
      .from(comments)
      .where(eq(comments.artworkId, req.params.id))
      .orderBy(desc(comments.createdAt), desc(sql`rowid`))
      .all();
    res.json(rows);
  });

  // --- reports / moderation ------------------------------------------------
  const createReportSchema = z.object({ reporterId: z.string().min(1), reason: z.string().min(1).max(1000) });

  router.post("/artworks/:id/reports", (req, res) => {
    const parsed = createReportSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
    if (!artworkExists(db, req.params.id)) return res.status(404).json({ error: `no artwork ${req.params.id}` });

    const id = `rpt_${randomUUID().replace(/-/g, "").slice(0, 16)}`;
    db.insert(reports)
      .values({
        id,
        reporterId: parsed.data.reporterId,
        artworkId: req.params.id,
        reason: parsed.data.reason,
        status: "PENDING",
        createdAt: new Date(),
      })
      .run();
    res.status(201).json({ id, status: "PENDING" });
  });

  // Not role-gated here (this service doesn't know about roles) --
  // api-gateway's community router only mounts this under a
  // MODERATOR/ADMIN-only path. Calling this endpoint directly bypasses
  // that, same trust-boundary note as this file's module doc.
  router.get("/moderation/reports", (req, res) => {
    const status = typeof req.query.status === "string" ? req.query.status : "PENDING";
    const rows = db.select().from(reports).where(eq(reports.status, status)).orderBy(desc(reports.createdAt)).all();
    res.json(rows);
  });

  const updateReportSchema = z.object({ status: z.enum(["RESOLVED", "DISMISSED"]) });

  router.patch("/moderation/reports/:id", (req, res) => {
    const parsed = updateReportSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });

    const existing = db.select().from(reports).where(eq(reports.id, req.params.id)).get();
    if (!existing) return res.status(404).json({ error: `no report ${req.params.id}` });
    if (existing.status !== "PENDING") {
      return res.status(409).json({ error: `report ${req.params.id} already ${existing.status}` });
    }

    db.update(reports).set({ status: parsed.data.status }).where(eq(reports.id, req.params.id)).run();
    res.json({ id: req.params.id, status: parsed.data.status });
  });

  // --- feed ----------------------------------------------------------------
  router.get("/feed", (req, res) => {
    const type = typeof req.query.type === "string" ? req.query.type : "latest";
    const limit = Math.min(Number(req.query.limit) || 20, 100);

    if (type === "following") {
      const userId = typeof req.query.userId === "string" ? req.query.userId : undefined;
      if (!userId) return res.status(400).json({ error: "userId is required for the 'following' feed" });

      const followedIds = db.select({ creatorId: follows.creatorId }).from(follows).where(eq(follows.followerId, userId)).all();
      if (followedIds.length === 0) return res.json([]);

      const ids = followedIds.map((f) => f.creatorId);
      const rows = db
        .select()
        .from(artworks)
        .where(and(eq(artworks.visibility, "public"), eq(artworks.status, "PUBLISHED"), sql`${artworks.creatorId} in ${ids}`))
        .orderBy(desc(artworks.publishedAt))
        .limit(limit)
        .all();
      return res.json(rows);
    }

    if (type === "popular") {
      // Popularity = like count, computed here rather than denormalized on
      // the artworks row -- this feed isn't latency-critical enough yet to
      // need a maintained counter, and a real one would need to handle
      // unlikes too (see likes' DELETE route above).
      const rows = db
        .select({ artwork: artworks, likeCount: sql<number>`count(${likes.userId})`.as("like_count") })
        .from(artworks)
        .leftJoin(likes, eq(likes.artworkId, artworks.id))
        .where(and(eq(artworks.visibility, "public"), eq(artworks.status, "PUBLISHED")))
        .groupBy(artworks.id)
        .orderBy(desc(sql`like_count`))
        .limit(limit)
        .all();
      return res.json(rows.map((r) => ({ ...r.artwork, likeCount: r.likeCount })));
    }

    // "latest" (default)
    const rows = db
      .select()
      .from(artworks)
      .where(and(eq(artworks.visibility, "public"), eq(artworks.status, "PUBLISHED")))
      .orderBy(desc(artworks.publishedAt))
      .limit(limit)
      .all();
    res.json(rows);
  });

  return router;
}

function artworkExists(db: Db, id: string): boolean {
  return db.select({ id: artworks.id }).from(artworks).where(eq(artworks.id, id)).get() !== undefined;
}
