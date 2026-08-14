import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import * as community from "../api/community";
import { api } from "../api/client";
import type { Artwork, FeedType } from "../api/types";
import { ArtworkGrid } from "../components/ArtworkGrid";

const TABS: Array<{ type: FeedType; label: string }> = [
  { type: "latest", label: "최신" },
  { type: "popular", label: "인기" },
  { type: "following", label: "팔로잉" },
];

export function FeedPage() {
  const [type, setType] = useState<FeedType>("latest");
  const [searchParams, setSearchParams] = useSearchParams();
  const [searchInput, setSearchInput] = useState(searchParams.get("tag") ?? "");
  const activeTag = searchParams.get("tag") ?? "";

  const feedQuery = useQuery({
    queryKey: ["feed", type],
    queryFn: () => community.getFeed(type),
    enabled: !activeTag,
  });

  // Tag search feature (2026-08-14): a dedicated global search across every
  // creator's public work, not scoped to the latest/popular/following tabs
  // above -- see api-gateway's GET /artworks/search doc for why this is
  // always publicOnly regardless of who's asking.
  const searchQuery = useQuery({
    queryKey: ["artworkTagSearch", activeTag],
    queryFn: () => api.get<Artwork[]>(`/artworks/search?tag=${encodeURIComponent(activeTag)}`),
    enabled: !!activeTag,
  });

  const runSearch = (tag: string) => {
    const trimmed = tag.trim();
    if (trimmed) setSearchParams({ tag: trimmed });
    else setSearchParams({});
  };

  return (
    <div className="mx-auto max-w-6xl">
      <form
        className="mb-4 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          runSearch(searchInput);
        }}
      >
        <input
          type="text"
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          placeholder="태그로 검색 (예: 수채화)"
          className="flex-1 rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm"
        />
        <button type="submit" className="btn-secondary">
          검색
        </button>
        {activeTag && (
          <button
            type="button"
            className="btn-secondary"
            onClick={() => {
              setSearchInput("");
              setSearchParams({});
            }}
          >
            초기화
          </button>
        )}
      </form>

      {!activeTag && (
        <div className="mb-6 flex gap-1 border-b border-neutral-800">
          {TABS.map((tab) => (
            <button
              key={tab.type}
              onClick={() => setType(tab.type)}
              className={`relative px-4 py-3 text-sm font-semibold transition-colors ${
                type === tab.type ? "text-neutral-50" : "text-neutral-500 hover:text-neutral-200"
              }`}
            >
              {tab.label}
              {type === tab.type && <span className="absolute inset-x-3 -bottom-px h-0.5 rounded-full bg-brand-500" />}
            </button>
          ))}
        </div>
      )}

      {activeTag && (
        <p className="mb-4 text-sm text-neutral-400">
          <span className="text-neutral-200">#{activeTag}</span> 태그 검색 결과
        </p>
      )}

      {activeTag ? (
        <>
          {searchQuery.error && <p className="py-16 text-center text-sm text-red-400">검색에 실패했습니다.</p>}
          {!searchQuery.error && (
            <ArtworkGrid
              artworks={searchQuery.isLoading ? undefined : searchQuery.data}
              emptyMessage={`"${activeTag}" 태그를 가진 공개 작품이 없습니다.`}
            />
          )}
        </>
      ) : (
        <>
          {feedQuery.error && <p className="py-16 text-center text-sm text-red-400">피드를 불러오지 못했습니다.</p>}
          {!feedQuery.error && (
            <ArtworkGrid
              artworks={feedQuery.isLoading ? undefined : feedQuery.data}
              emptyMessage={type === "following" ? "팔로우한 창작자의 공개 작품이 아직 없습니다." : "공개된 작품이 아직 없습니다."}
            />
          )}
        </>
      )}
    </div>
  );
}
