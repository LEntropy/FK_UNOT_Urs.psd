import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import * as community from "../api/community";
import { getUsersBatch } from "../api/users";
import { Avatar } from "../components/Avatar";
import { FollowButton } from "../components/FollowButton";
import { BackIcon } from "../components/icons";

/** Follow list page (2026-08-14, new) -- the "팔로우 리스트나 이런 것도
 * 없고" gap. Shared by /creators/:id/followers and /creators/:id/following;
 * `kind` picks which list + which backend list endpoint to call. The list
 * endpoints only return raw user ids (asset-service owns the follow graph,
 * not display identity) -- resolved to handles/avatars via one batched
 * GET /users?ids= call, same pattern ArtworkGrid uses for thumbnails. */
export function FollowListPage({ kind }: { kind: "followers" | "following" }) {
  const { id } = useParams<{ id: string }>();

  const { data: edges, isLoading } = useQuery({
    queryKey: [kind, id],
    queryFn: () => (kind === "followers" ? community.listFollowers(id!) : community.listFollowing(id!)),
    enabled: Boolean(id),
  });

  const userIds = (edges ?? []).map((e) => e.userId);
  const { data: profiles } = useQuery({
    queryKey: ["usersBatch", userIds.join(",")],
    queryFn: () => getUsersBatch(userIds),
    enabled: userIds.length > 0,
  });

  return (
    <div className="mx-auto max-w-md">
      <div className="mb-4 flex items-center gap-3">
        <Link to={`/creators/${id}`} className="rounded-full p-1.5 hover:bg-neutral-900">
          <BackIcon className="h-5 w-5" />
        </Link>
        <h1 className="text-lg font-bold text-neutral-50">{kind === "followers" ? "팔로워" : "팔로잉"}</h1>
      </div>

      {isLoading && <p className="py-12 text-center text-sm text-neutral-500">불러오는 중...</p>}
      {!isLoading && edges?.length === 0 && (
        <p className="py-12 text-center text-sm text-neutral-500">
          {kind === "followers" ? "아직 팔로워가 없습니다." : "아직 팔로우한 창작자가 없습니다."}
        </p>
      )}

      <ul className="flex flex-col divide-y divide-neutral-900">
        {edges?.map((edge) => {
          const profile = profiles?.find((p) => p.id === edge.userId);
          return (
            <li key={edge.userId} className="flex items-center gap-3 py-3">
              <Link to={`/creators/${edge.userId}`} className="flex min-w-0 flex-1 items-center gap-3">
                <Avatar seed={edge.userId} label={profile?.displayName || profile?.handle || edge.userId} avatarUri={profile?.avatarUri} size="sm" />
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-neutral-100">
                    {profile?.displayName || profile?.handle || "..."}
                  </p>
                  {profile && <p className="truncate text-xs text-neutral-500">@{profile.handle}</p>}
                </div>
              </Link>
              <FollowButton creatorId={edge.userId} className="!px-3 !py-1 !text-xs" />
            </li>
          );
        })}
      </ul>
    </div>
  );
}
