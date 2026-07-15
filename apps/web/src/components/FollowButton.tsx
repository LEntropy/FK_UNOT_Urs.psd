import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as community from "../api/community";
import { useAuthStore } from "../store/auth";

/** Same "no am-I-following endpoint" limitation as LikeButton -- see its
 * comment for why this starts every load assuming "not following yet". */
export function FollowButton({ creatorId }: { creatorId: string }) {
  const currentUserId = useAuthStore((s) => s.user?.id);
  const queryClient = useQueryClient();
  const [followingThisSession, setFollowingThisSession] = useState(false);

  const { data } = useQuery({
    queryKey: ["followerCount", creatorId],
    queryFn: () => community.followerCount(creatorId),
  });

  const toggle = useMutation({
    mutationFn: () => (followingThisSession ? community.unfollow(creatorId) : community.follow(creatorId)),
    onSuccess: () => {
      setFollowingThisSession((v) => !v);
      queryClient.invalidateQueries({ queryKey: ["followerCount", creatorId] });
    },
  });

  if (currentUserId === creatorId) return null; // can't follow yourself, matches api-gateway's own check

  return (
    <button
      onClick={() => toggle.mutate()}
      disabled={toggle.isPending}
      className={`rounded border px-3 py-1.5 text-sm ${
        followingThisSession
          ? "border-neutral-600 bg-neutral-800 text-neutral-200"
          : "border-blue-700 bg-blue-950/40 text-blue-300"
      }`}
    >
      {followingThisSession ? "팔로잉" : "+ 팔로우"} {typeof data?.count === "number" ? `(${data.count})` : ""}
    </button>
  );
}
