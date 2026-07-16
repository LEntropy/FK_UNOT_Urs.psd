import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import * as community from "../api/community";
import type { FeedType } from "../api/types";
import { ArtworkImage } from "../components/ArtworkImage";

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
    <div className="mx-auto mt-8 max-w-5xl">
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

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
        {data?.map((artwork) => (
          <Link
            key={artwork.id}
            to={`/artworks/${artwork.id}`}
            className="group flex flex-col overflow-hidden rounded border border-neutral-800 hover:border-neutral-600"
          >
            <ArtworkImage
              artworkId={artwork.id}
              hasVariants={artwork.assetVersions.length > 0}
              variant="thumbnail"
              className="aspect-square w-full object-cover"
            />
            <div className="flex flex-col gap-0.5 px-2 py-2">
              <span className="truncate text-sm font-medium">{artwork.title}</span>
              <div className="flex items-center justify-between text-xs text-neutral-500">
                <span className="truncate">@{artwork.creatorId}</span>
                {typeof artwork.likeCount === "number" && <span>♡ {artwork.likeCount}</span>}
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
