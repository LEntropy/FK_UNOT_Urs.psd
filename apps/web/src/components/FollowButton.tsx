import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as community from "../api/community";
import { useAuthStore } from "../store/auth";

/** 2026-08-14: now backed by a real "am I following" check
 * (community.followStatus) instead of assuming "not following" on every
 * page load -- see asset-service routes/community.ts's follow-status
 * route doc for the backend side of this fix. */
export function FollowButton({ creatorId, className }: { creatorId: string; className?: string }) {
  const currentUserId = useAuthStore((s) => s.user?.id);
  const queryClient = useQueryClient();

  const { data: countData } = useQuery({
    queryKey: ["followerCount", creatorId],
    queryFn: () => community.followerCount(creatorId),
  });
  const { data: statusData } = useQuery({
    queryKey: ["followStatus", creatorId],
    queryFn: () => community.followStatus(creatorId),
    enabled: currentUserId !== creatorId,
  });

  const following = statusData?.following ?? false;

  const toggle = useMutation({
    mutationFn: () => (following ? community.unfollow(creatorId) : community.follow(creatorId)),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["followerCount", creatorId] });
      queryClient.invalidateQueries({ queryKey: ["followStatus", creatorId] });
    },
  });

  if (currentUserId === creatorId) return null; // can't follow yourself, matches api-gateway's own check

  return (
    <button
      onClick={() => toggle.mutate()}
      disabled={toggle.isPending}
      className={`${following ? "btn-secondary" : "btn-primary"} ${className ?? ""}`}
    >
      {following ? "팔로잉" : "팔로우"} {typeof countData?.count === "number" ? `(${countData.count})` : ""}
    </button>
  );
}
