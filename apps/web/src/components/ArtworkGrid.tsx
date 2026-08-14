import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import type { Artwork } from "../api/types";
import { getRenderUrlsBatch } from "../api/delivery";
import { ArtworkImage } from "./ArtworkImage";
import { HeartIcon } from "./icons";

const STATUS_LABEL: Record<Artwork["status"], string> = {
  UPLOADED: "업로드됨",
  PROTECTING: "보호 처리 중",
  REGISTERING: "온체인 등록 중",
  PUBLISHED: "공개됨",
  FAILED: "실패",
};

/**
 * Shared Pixiv-style browsing grid (2026-08-14 redesign) -- used by
 * FeedPage/GalleryPage/ProfilePage instead of each page hand-rolling its
 * own <div className="grid ..."> + <ArtworkImage> loop. Batches every
 * visible card's thumbnail URL into one request (see api/delivery.ts's
 * getRenderUrlsBatch doc for why that's the real fix for "피드 로딩이
 * 느리다" -- N cards used to mean N+N round trips competing for the
 * browser's ~6-connections-per-origin limit before this).
 */
export function ArtworkGrid({
  artworks,
  showStatus = false,
  emptyMessage = "아직 작품이 없습니다.",
}: {
  artworks: Array<Artwork & { likeCount?: number }> | undefined;
  showStatus?: boolean;
  emptyMessage?: string;
}) {
  const renderableIds = (artworks ?? []).filter((a) => a.assetVersions.length > 0).map((a) => a.id);

  const { data: urls } = useQuery({
    queryKey: ["renderUrlsBatch", renderableIds.join(",")],
    queryFn: () => getRenderUrlsBatch(renderableIds, "thumbnail"),
    enabled: renderableIds.length > 0,
    staleTime: 4 * 60 * 1000,
  });

  if (artworks && artworks.length === 0) {
    return <p className="py-16 text-center text-sm text-neutral-500">{emptyMessage}</p>;
  }

  return (
    <div className="grid grid-cols-2 gap-1 sm:grid-cols-3 sm:gap-2 md:grid-cols-4 lg:grid-cols-5">
      {(artworks ?? Array.from({ length: 10 })).map((artwork, i) =>
        artwork ? (
          <Link
            key={artwork.id}
            to={`/artworks/${artwork.id}`}
            className="group relative aspect-square overflow-hidden rounded-xl bg-neutral-900"
          >
            <ArtworkImage
              artworkId={artwork.id}
              hasVariants={artwork.assetVersions.length > 0}
              variant="thumbnail"
              batchMode
              preloadedSrc={urls ? (urls[artwork.id] ?? null) : undefined}
              className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-105"
            />
            {showStatus && artwork.status !== "PUBLISHED" && (
              <span className="absolute right-2 top-2 rounded-full bg-neutral-950/80 px-2 py-0.5 text-[11px] font-medium text-neutral-200 backdrop-blur">
                {STATUS_LABEL[artwork.status]}
              </span>
            )}
            <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-2 bg-gradient-to-t from-black/80 via-black/20 to-transparent p-2.5 opacity-0 transition-opacity group-hover:opacity-100">
              <span className="truncate text-xs font-medium text-white">{artwork.title}</span>
              {typeof artwork.likeCount === "number" && (
                <span className="flex shrink-0 items-center gap-1 text-xs text-white/90">
                  <HeartIcon className="h-3.5 w-3.5" />
                  {artwork.likeCount}
                </span>
              )}
            </div>
          </Link>
        ) : (
          <div key={i} className="skeleton aspect-square rounded-xl" />
        ),
      )}
    </div>
  );
}
