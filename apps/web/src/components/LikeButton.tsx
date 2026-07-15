import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as community from "../api/community";

/**
 * There's no "did the current user already like this" endpoint on the
 * backend (community.ts's likes table has no such query) -- so this
 * starts every page load assuming "not liked yet" rather than guessing.
 * Liking/unliking is idempotent server-side either way (POST twice is a
 * no-op, DELETE when not liked is a no-op), so this never desyncs the
 * actual count, it just can't show "you already liked this" across a
 * reload. Documented here rather than silently pretending it's tracked.
 */
export function LikeButton({ artworkId }: { artworkId: string }) {
  const queryClient = useQueryClient();
  const [likedThisSession, setLikedThisSession] = useState(false);

  const { data } = useQuery({
    queryKey: ["likeCount", artworkId],
    queryFn: () => community.likeCount(artworkId),
  });

  const toggle = useMutation({
    mutationFn: () => (likedThisSession ? community.unlike(artworkId) : community.like(artworkId)),
    onSuccess: () => {
      setLikedThisSession((v) => !v);
      queryClient.invalidateQueries({ queryKey: ["likeCount", artworkId] });
    },
  });

  return (
    <button
      onClick={() => toggle.mutate()}
      disabled={toggle.isPending}
      className={`rounded border px-3 py-1.5 text-sm ${
        likedThisSession
          ? "border-pink-700 bg-pink-950/40 text-pink-300"
          : "border-neutral-700 hover:bg-neutral-900"
      }`}
    >
      {likedThisSession ? "♥ 좋아요 취소" : "♡ 좋아요"} {typeof data?.count === "number" ? `(${data.count})` : ""}
    </button>
  );
}
