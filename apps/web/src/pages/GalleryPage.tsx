import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Artwork } from "../api/types";
import { ArtworkImage } from "../components/ArtworkImage";

const STATUS_LABEL: Record<Artwork["status"], string> = {
  UPLOADED: "업로드됨",
  PROTECTING: "보호 처리 중",
  REGISTERING: "온체인 등록 중",
  PUBLISHED: "공개됨",
  FAILED: "실패",
};

const STATUS_COLOR: Record<Artwork["status"], string> = {
  UPLOADED: "bg-neutral-800 text-neutral-300",
  PROTECTING: "bg-neutral-800 text-neutral-300",
  REGISTERING: "bg-neutral-800 text-neutral-300",
  PUBLISHED: "bg-green-900 text-green-300",
  FAILED: "bg-red-900 text-red-300",
};

export function GalleryPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["artworks"],
    // Still processing artworks are polled every few seconds elsewhere
    // (ArtworkDetailPage) -- here a light poll is enough to move a card
    // from "처리 중" to a real thumbnail without a manual refresh.
    queryFn: () => api.get<Artwork[]>("/artworks"),
    refetchInterval: 5000,
  });

  if (isLoading) return <p className="mt-12 text-center text-neutral-400">불러오는 중...</p>;
  if (error) return <p className="mt-12 text-center text-red-400">작품 목록을 불러오지 못했습니다.</p>;

  return (
    <div className="mx-auto mt-8 max-w-5xl">
      <h1 className="mb-6 text-2xl font-semibold">내 작품</h1>
      {data?.length === 0 && <p className="text-neutral-400">아직 업로드한 작품이 없습니다.</p>}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
        {data?.map((artwork) => (
          <Link
            key={artwork.id}
            to={`/artworks/${artwork.id}`}
            className="group relative flex flex-col overflow-hidden rounded border border-neutral-800 hover:border-neutral-600"
          >
            <ArtworkImage
              artworkId={artwork.id}
              hasVariants={artwork.assetVersions.length > 0}
              variant="thumbnail"
              className="aspect-square w-full object-cover"
            />
            <span
              className={`absolute right-2 top-2 rounded px-1.5 py-0.5 text-[11px] font-medium ${STATUS_COLOR[artwork.status]}`}
            >
              {STATUS_LABEL[artwork.status]}
            </span>
            <div className="px-2 py-2">
              <span className="truncate text-sm font-medium">{artwork.title}</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
