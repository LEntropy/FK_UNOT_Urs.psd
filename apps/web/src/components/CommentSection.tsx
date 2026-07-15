import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as community from "../api/community";

export function CommentSection({ artworkId }: { artworkId: string }) {
  const queryClient = useQueryClient();
  const [body, setBody] = useState("");

  const { data: comments } = useQuery({
    queryKey: ["comments", artworkId],
    queryFn: () => community.listComments(artworkId),
  });

  const post = useMutation({
    mutationFn: () => community.postComment(artworkId, body),
    onSuccess: () => {
      setBody("");
      queryClient.invalidateQueries({ queryKey: ["comments", artworkId] });
    },
  });

  return (
    <div className="mt-6">
      <h2 className="mb-3 text-sm font-medium text-neutral-300">댓글 {comments?.length ?? 0}개</h2>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (body.trim()) post.mutate();
        }}
        className="mb-4 flex gap-2"
      >
        <input
          value={body}
          onChange={(e) => setBody(e.target.value)}
          placeholder="댓글을 입력하세요"
          maxLength={2000}
          className="flex-1 rounded border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-sm"
        />
        <button
          type="submit"
          disabled={post.isPending || !body.trim()}
          className="rounded bg-neutral-100 px-3 py-1.5 text-sm font-medium text-neutral-900 disabled:opacity-50"
        >
          등록
        </button>
      </form>
      <ul className="flex flex-col gap-2">
        {comments?.map((c) => (
          <li key={c.id} className="rounded border border-neutral-800 px-3 py-2 text-sm">
            <span className="text-neutral-500">{c.userId}</span>
            <p className="mt-1">{c.body}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
