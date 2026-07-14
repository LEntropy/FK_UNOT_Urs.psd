import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Artwork } from "../api/types";

const STATUS_LABEL: Record<Artwork["status"], string> = {
  UPLOADED: "업로드됨",
  PROTECTING: "보호 처리 중",
  REGISTERING: "온체인 등록 중",
  PUBLISHED: "공개됨",
  FAILED: "실패",
};

export function GalleryPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["artworks"],
    queryFn: () => api.get<Artwork[]>("/artworks"),
  });

  if (isLoading) return <p className="mt-12 text-center text-neutral-400">불러오는 중...</p>;
  if (error) return <p className="mt-12 text-center text-red-400">작품 목록을 불러오지 못했습니다.</p>;

  return (
    <div className="mx-auto mt-8 max-w-3xl">
      <h1 className="mb-6 text-2xl font-semibold">내 작품</h1>
      {data?.length === 0 && <p className="text-neutral-400">아직 업로드한 작품이 없습니다.</p>}
      <ul className="flex flex-col gap-3">
        {data?.map((artwork) => (
          <li key={artwork.id}>
            <Link
              to={`/artworks/${artwork.id}`}
              className="flex items-center justify-between rounded border border-neutral-800 px-4 py-3 hover:bg-neutral-900"
            >
              <span className="font-medium">{artwork.title}</span>
              <span className="text-sm text-neutral-400">{STATUS_LABEL[artwork.status]}</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
