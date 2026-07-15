import { Router } from "express";
import { z } from "zod";
import { requireAuth } from "../middleware/requireAuth.js";
import * as community from "../clients/community.js";
import { AssetServiceError } from "../clients/assetService.js";

/**
 * Thin authenticated proxy in front of asset-service's community routes,
 * same pattern as artworks.ts: the frontend only ever talks to this router,
 * userId/reporterId always come from the verified JWT, never the request
 * body -- so a caller can't like/follow/report as someone else.
 */
export function communityRouter(): Router {
  const router = Router();
  router.use(requireAuth);

  // --- likes ---------------------------------------------------------
  router.post("/artworks/:id/likes", async (req, res) => {
    try {
      await community.like(req.params.id, req.user!.sub);
      res.status(204).send();
    } catch (err) {
      forwardError(err, res);
    }
  });
  router.delete("/artworks/:id/likes", async (req, res) => {
    try {
      await community.unlike(req.params.id, req.user!.sub);
      res.status(204).send();
    } catch (err) {
      forwardError(err, res);
    }
  });
  router.get("/artworks/:id/likes/count", async (req, res) => {
    try {
      res.json(await community.likeCount(req.params.id));
    } catch (err) {
      forwardError(err, res);
    }
  });

  // --- bookmarks -------------------------------------------------------
  const bookmarkSchema = z.object({ collectionId: z.string().min(1).optional() });
  router.post("/artworks/:id/bookmarks", async (req, res) => {
    const parsed = bookmarkSchema.safeParse(req.body ?? {});
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
    try {
      await community.bookmark(req.params.id, req.user!.sub, parsed.data.collectionId);
      res.status(204).send();
    } catch (err) {
      forwardError(err, res);
    }
  });
  router.delete("/artworks/:id/bookmarks", async (req, res) => {
    try {
      await community.unbookmark(req.params.id, req.user!.sub);
      res.status(204).send();
    } catch (err) {
      forwardError(err, res);
    }
  });
  router.get("/me/bookmarks", async (req, res) => {
    try {
      res.json(await community.listBookmarks(req.user!.sub));
    } catch (err) {
      forwardError(err, res);
    }
  });

  // --- collections -----------------------------------------------------
  const createCollectionSchema = z.object({ name: z.string().min(1), isPublic: z.boolean().optional() });
  router.post("/collections", async (req, res) => {
    const parsed = createCollectionSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
    try {
      res.status(201).json(await community.createCollection(req.user!.sub, parsed.data.name, parsed.data.isPublic));
    } catch (err) {
      forwardError(err, res);
    }
  });
  router.get("/me/collections", async (req, res) => {
    try {
      res.json(await community.listCollections(req.user!.sub));
    } catch (err) {
      forwardError(err, res);
    }
  });

  // --- follows -----------------------------------------------------------
  router.post("/users/:creatorId/follow", async (req, res) => {
    try {
      await community.follow(req.params.creatorId, req.user!.sub);
      res.status(204).send();
    } catch (err) {
      forwardError(err, res);
    }
  });
  router.delete("/users/:creatorId/follow", async (req, res) => {
    try {
      await community.unfollow(req.params.creatorId, req.user!.sub);
      res.status(204).send();
    } catch (err) {
      forwardError(err, res);
    }
  });
  router.get("/users/:creatorId/followers/count", async (req, res) => {
    try {
      res.json(await community.followerCount(req.params.creatorId));
    } catch (err) {
      forwardError(err, res);
    }
  });

  // --- comments ----------------------------------------------------------
  const createCommentSchema = z.object({ body: z.string().min(1).max(2000) });
  router.post("/artworks/:id/comments", async (req, res) => {
    const parsed = createCommentSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
    try {
      res.status(201).json(await community.createComment(req.params.id, req.user!.sub, parsed.data.body));
    } catch (err) {
      forwardError(err, res);
    }
  });
  router.get("/artworks/:id/comments", async (req, res) => {
    try {
      res.json(await community.listComments(req.params.id));
    } catch (err) {
      forwardError(err, res);
    }
  });

  // --- reports -------------------------------------------------------------
  const createReportSchema = z.object({ reason: z.string().min(1).max(1000) });
  router.post("/artworks/:id/reports", async (req, res) => {
    const parsed = createReportSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
    try {
      res.status(201).json(await community.createReport(req.params.id, req.user!.sub, parsed.data.reason));
    } catch (err) {
      forwardError(err, res);
    }
  });

  // --- moderation (MODERATOR/ADMIN only) ------------------------------------
  const requireModerator = (req: import("express").Request, res: import("express").Response, next: import("express").NextFunction) => {
    if (req.user!.role !== "MODERATOR" && req.user!.role !== "ADMIN") {
      return res.status(403).json({ error: "moderator or admin role required" });
    }
    next();
  };

  router.get("/moderation/reports", requireModerator, async (req, res) => {
    const status = typeof req.query.status === "string" ? req.query.status : undefined;
    try {
      res.json(await community.listModerationQueue(status));
    } catch (err) {
      forwardError(err, res);
    }
  });
  const resolveReportSchema = z.object({ status: z.enum(["RESOLVED", "DISMISSED"]) });
  router.patch("/moderation/reports/:id", requireModerator, async (req, res) => {
    const parsed = resolveReportSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
    try {
      res.json(await community.resolveReport(req.params.id, parsed.data.status));
    } catch (err) {
      forwardError(err, res);
    }
  });

  // --- feed ----------------------------------------------------------------
  router.get("/feed", async (req, res) => {
    const type = req.query.type === "following" || req.query.type === "popular" ? req.query.type : "latest";
    const limit = req.query.limit ? Number(req.query.limit) : undefined;
    try {
      res.json(await community.getFeed(type, type === "following" ? req.user!.sub : undefined, limit));
    } catch (err) {
      forwardError(err, res);
    }
  });

  return router;
}

function forwardError(err: unknown, res: import("express").Response) {
  if (err instanceof AssetServiceError) {
    return res.status(err.status).json(err.body);
  }
  res.status(502).json({ error: "asset-service unreachable" });
}
