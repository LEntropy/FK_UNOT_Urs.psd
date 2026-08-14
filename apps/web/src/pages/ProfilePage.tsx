import { useParams, Link, Navigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { getUser } from "../api/users";
import * as community from "../api/community";
import type { Artwork } from "../api/types";
import { useAuthStore } from "../store/auth";
import { Avatar } from "../components/Avatar";
import { FollowButton } from "../components/FollowButton";
import { ArtworkGrid } from "../components/ArtworkGrid";
import { SettingsIcon } from "../components/icons";

/** Public creator profile (2026-08-14, new) -- avatar/handle/bio header,
 * follower/following counts (linking to FollowListPage), and a Pixiv-style
 * grid of that creator's published public work. Own-profile view swaps the
 * FollowButton for a link to /settings, matching where X/Pixiv put the
 * "edit profile" affordance. */
export function ProfilePage() {
  const { id } = useParams<{ id: string }>();
  const currentUserId = useAuthStore((s) => s.user?.id);
  const isOwnProfile = id === currentUserId;

  const { data: profile, isLoading, error } = useQuery({
    queryKey: ["user", id],
    queryFn: () => getUser(id!),
    enabled: Boolean(id),
  });

  const { data: artworks } = useQuery({
    queryKey: ["artworksByCreator", id],
    queryFn: () => api.get<Artwork[]>(`/artworks/by-creator/${id}`),
    enabled: Boolean(id),
  });

  const { data: followerCount } = useQuery({
    queryKey: ["followerCount", id],
    queryFn: () => community.followerCount(id!),
    enabled: Boolean(id),
  });
  const { data: followingList } = useQuery({
    queryKey: ["following", id],
    queryFn: () => community.listFollowing(id!),
    enabled: Boolean(id),
  });

  if (!id) return <Navigate to="/" replace />;
  if (isLoading) return <p className="mt-12 text-center text-sm text-neutral-400">불러오는 중...</p>;
  if (error || !profile) return <p className="mt-12 text-center text-sm text-red-400">사용자를 찾을 수 없습니다.</p>;

  return (
    <div className="mx-auto max-w-6xl">
      <div className="card mb-8 flex flex-col gap-5 p-6 sm:flex-row sm:items-start">
        <Avatar seed={profile.id} label={profile.displayName || profile.handle} avatarUri={profile.avatarUri} size="xl" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="min-w-0">
              <h1 className="truncate text-xl font-bold text-neutral-50">{profile.displayName || profile.handle}</h1>
              <p className="text-sm text-neutral-500">@{profile.handle}</p>
            </div>
            {isOwnProfile ? (
              <Link to="/settings" className="btn-secondary">
                <SettingsIcon className="h-4 w-4" />
                프로필 편집
              </Link>
            ) : (
              <FollowButton creatorId={id} />
            )}
          </div>
          {profile.bio && <p className="mt-3 whitespace-pre-wrap text-sm text-neutral-300">{profile.bio}</p>}
          <div className="mt-4 flex gap-5 text-sm">
            <Link to={`/creators/${id}/followers`} className="text-neutral-300 hover:underline">
              <span className="font-semibold text-neutral-50">{followerCount?.count ?? 0}</span>{" "}
              <span className="text-neutral-500">팔로워</span>
            </Link>
            <Link to={`/creators/${id}/following`} className="text-neutral-300 hover:underline">
              <span className="font-semibold text-neutral-50">{followingList?.length ?? 0}</span>{" "}
              <span className="text-neutral-500">팔로잉</span>
            </Link>
            <span className="text-neutral-300">
              <span className="font-semibold text-neutral-50">{artworks?.length ?? 0}</span>{" "}
              <span className="text-neutral-500">작품</span>
            </span>
          </div>
        </div>
      </div>

      <ArtworkGrid artworks={artworks} emptyMessage="아직 공개한 작품이 없습니다." />
    </div>
  );
}
