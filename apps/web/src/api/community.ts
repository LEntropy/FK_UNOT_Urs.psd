import { api } from "./client";
import type { Artwork, Comment, FeedType, Report } from "./types";

export const getFeed = (type: FeedType) =>
  api.get<Array<Artwork & { likeCount?: number }>>(`/feed?type=${type}`);

export const like = (artworkId: string) => api.post<void>(`/artworks/${artworkId}/likes`);
export const unlike = (artworkId: string) => api.delete<void>(`/artworks/${artworkId}/likes`);
export const likeCount = (artworkId: string) => api.get<{ count: number }>(`/artworks/${artworkId}/likes/count`);

export const follow = (creatorId: string) => api.post<void>(`/users/${creatorId}/follow`);
export const unfollow = (creatorId: string) => api.delete<void>(`/users/${creatorId}/follow`);
export const followerCount = (creatorId: string) =>
  api.get<{ count: number }>(`/users/${creatorId}/followers/count`);

export const listComments = (artworkId: string) => api.get<Comment[]>(`/artworks/${artworkId}/comments`);
export const postComment = (artworkId: string, body: string) =>
  api.post<{ id: string }>(`/artworks/${artworkId}/comments`, { body });

export const reportArtwork = (artworkId: string, reason: string) =>
  api.post<{ id: string; status: string }>(`/artworks/${artworkId}/reports`, { reason });

export const listModerationQueue = (status = "PENDING") =>
  api.get<Report[]>(`/moderation/reports?status=${status}`);
export const resolveReport = (reportId: string, status: "RESOLVED" | "DISMISSED") =>
  api.patch<{ id: string; status: string }>(`/moderation/reports/${reportId}`, { status });
