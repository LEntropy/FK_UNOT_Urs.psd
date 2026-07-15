import { env } from "../env.js";
import { AssetServiceError } from "./assetService.js";

/**
 * Thin pass-through to asset-service's community routes (routes/community.ts
 * there) -- same pattern as assetService.ts: api-gateway's job is auth +
 * injecting the caller's identity, not reshaping the payload.
 */

async function call(path: string, init: RequestInit = {}) {
  const res = await fetch(`${env.ASSET_SERVICE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init.headers },
  });
  if (res.status === 204) return undefined;
  const body = await res.json();
  if (!res.ok) throw new AssetServiceError(res.status, body);
  return body;
}

export const like = (artworkId: string, userId: string) =>
  call(`/artworks/${encodeURIComponent(artworkId)}/likes`, { method: "POST", body: JSON.stringify({ userId }) });
export const unlike = (artworkId: string, userId: string) =>
  call(`/artworks/${encodeURIComponent(artworkId)}/likes`, { method: "DELETE", body: JSON.stringify({ userId }) });
export const likeCount = (artworkId: string) => call(`/artworks/${encodeURIComponent(artworkId)}/likes/count`);

export const bookmark = (artworkId: string, userId: string, collectionId?: string) =>
  call(`/artworks/${encodeURIComponent(artworkId)}/bookmarks`, {
    method: "POST",
    body: JSON.stringify({ userId, collectionId }),
  });
export const unbookmark = (artworkId: string, userId: string) =>
  call(`/artworks/${encodeURIComponent(artworkId)}/bookmarks`, { method: "DELETE", body: JSON.stringify({ userId }) });
export const listBookmarks = (userId: string) => call(`/users/${encodeURIComponent(userId)}/bookmarks`);

export const createCollection = (userId: string, name: string, isPublic?: boolean) =>
  call("/collections", { method: "POST", body: JSON.stringify({ userId, name, isPublic }) });
export const listCollections = (userId?: string) =>
  call(userId ? `/collections?userId=${encodeURIComponent(userId)}` : "/collections");

export const follow = (creatorId: string, userId: string) =>
  call(`/users/${encodeURIComponent(creatorId)}/follow`, { method: "POST", body: JSON.stringify({ userId }) });
export const unfollow = (creatorId: string, userId: string) =>
  call(`/users/${encodeURIComponent(creatorId)}/follow`, { method: "DELETE", body: JSON.stringify({ userId }) });
export const followerCount = (creatorId: string) => call(`/users/${encodeURIComponent(creatorId)}/followers/count`);

export const createComment = (artworkId: string, userId: string, body: string) =>
  call(`/artworks/${encodeURIComponent(artworkId)}/comments`, {
    method: "POST",
    body: JSON.stringify({ userId, body }),
  });
export const listComments = (artworkId: string) => call(`/artworks/${encodeURIComponent(artworkId)}/comments`);

export const createReport = (artworkId: string, reporterId: string, reason: string) =>
  call(`/artworks/${encodeURIComponent(artworkId)}/reports`, {
    method: "POST",
    body: JSON.stringify({ reporterId, reason }),
  });
export const listModerationQueue = (status?: string) =>
  call(status ? `/moderation/reports?status=${encodeURIComponent(status)}` : "/moderation/reports");
export const resolveReport = (reportId: string, status: "RESOLVED" | "DISMISSED") =>
  call(`/moderation/reports/${encodeURIComponent(reportId)}`, { method: "PATCH", body: JSON.stringify({ status }) });

export const getFeed = (type: "following" | "popular" | "latest", userId?: string, limit?: number) => {
  const params = new URLSearchParams({ type });
  if (userId) params.set("userId", userId);
  if (limit) params.set("limit", String(limit));
  return call(`/feed?${params.toString()}`);
};
