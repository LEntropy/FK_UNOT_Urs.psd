import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type { Artwork } from "../api/types";
import { ArtworkGrid } from "../components/ArtworkGrid";

export function GalleryPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["artworks"],
    // Still processing artworks are polled every few seconds elsewhere
    // (ArtworkDetailPage) -- here a light poll is enough to move a card
    // from "처리 중" to a real thumbnail without a manual refresh.
    queryFn: () => api.get<Artwork[]>("/artworks"),
    refetchInterval: 5000,
  });

  if (error) return <p className="mt-12 text-center text-red-400">작품 목록을 불러오지 못했습니다.</p>;

  return (
    <div className="mx-auto max-w-6xl">
      <h1 className="mb-6 text-xl font-bold text-neutral-50">내 작품</h1>
      <ArtworkGrid artworks={isLoading ? undefined : data} showStatus emptyMessage="아직 업로드한 작품이 없습니다." />
    </div>
  );
}
