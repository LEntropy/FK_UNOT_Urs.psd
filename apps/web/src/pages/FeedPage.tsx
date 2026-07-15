import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import * as community from "../api/community";
import type { FeedType } from "../api/types";

const TABS: Array<{ type: FeedType; label: string }> = [
  { type: "latest", label: "최신" },
  { type: "popular", label: "인기" },
  { type: "following", label: "팔로잉" },
];

export function FeedPage() {
  const [type, setType] = useState<FeedType>("latest");

  const { data, isLoading, error } = useQuery({
    queryKey: ["feed", type],
    queryFn: () => community.getFeed(type),
  });

  return (
    <div className="mx-auto mt-8 max-w-3xl">
      <div className="mb-6 flex gap-2">
        {TABS.map((tab) => (
          <button
            key={tab.type}
            onClick={() => setType(tab.type)}
            className={`rounded px-3 py-1.5 text-sm ${
              type === tab.type ? "bg-neutral-100 text-neutral-900" : "border border-neutral-700 hover:bg-neutral-900"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {isLoading && <p className="text-neutral-400">불러오는 중...</p>}
      {error && <p className="text-red-400">피드를 불러오지 못했습니다.</p>}
      {data?.length === 0 && (
        <p className="text-neutral-400">
          {type === "following" ? "팔로우한 창작자의 공개 작품이 아직 없습니다." : "공개된 작품이 아직 없습니다."}
        </p>
      )}

      <ul className="flex flex-col gap-3">
        {data?.map((artwork) => (
          <li key={artwork.id}>
            <Link
              to={`/artworks/${artwork.id}`}
              className="flex items-center justify-between rounded border border-neutral-800 px-4 py-3 hover:bg-neutral-900"
            >
              <div>
                <span className="font-medium">{artwork.title}</span>
                <span className="ml-2 text-sm text-neutral-500">@{artwork.creatorId}</span>
              </div>
              {typeof artwork.likeCount === "number" && (
                <span className="text-sm text-neutral-400">♡ {artwork.likeCount}</span>
              )}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
