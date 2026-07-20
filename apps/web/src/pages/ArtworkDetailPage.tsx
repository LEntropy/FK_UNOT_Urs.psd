import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type { Artwork } from "../api/types";
import { ArtworkImage } from "../components/ArtworkImage";
import { LikeButton } from "../components/LikeButton";
import { FollowButton } from "../components/FollowButton";
import { CommentSection } from "../components/CommentSection";
import { ReportButton } from "../components/ReportButton";
import { ProtectionSummary } from "../components/ProtectionSummary";

const STATUS_LABEL: Record<Artwork["status"], string> = {
  UPLOADED: "업로드됨",
  PROTECTING: "보호 처리 중",
  REGISTERING: "온체인 등록 중",
  PUBLISHED: "공개됨",
  FAILED: "실패",
};

const TERMINAL_STATUSES: Artwork["status"][] = ["PUBLISHED", "FAILED"];

export function ArtworkDetailPage() {
  const { id } = useParams<{ id: string }>();

  const { data, isLoading, error } = useQuery({
    queryKey: ["artwork", id],
    queryFn: () => api.get<Artwork>(`/artworks/${id}`),
    enabled: Boolean(id),
    refetchInterval: (query) => (query.state.data && TERMINAL_STATUSES.includes(query.state.data.status) ? false : 2000),
  });

  if (isLoading) return <p className="mt-12 text-center text-neutral-400">불러오는 중...</p>;
  if (error || !data) return <p className="mt-12 text-center text-red-400">작품을 찾을 수 없습니다.</p>;

  const record = data.ownershipRecords?.[0];

  return (
    <div className="mx-auto mt-8 max-w-2xl">
      <h1 className="mb-1 text-2xl font-semibold">{data.title}</h1>
      <p className="mb-4 text-sm text-neutral-400">
        {data.id} · <span className="text-neutral-500">@{data.creatorId}</span>
      </p>

      <ArtworkImage
        artworkId={data.id}
        hasVariants={data.assetVersions.length > 0}
        className="mb-6 w-full rounded border border-neutral-800 object-contain"
      />

      <div className="mb-6 flex items-center gap-3">
        <StatusBadge status={data.status} />
        {!TERMINAL_STATUSES.includes(data.status) && (
          <span className="text-xs text-neutral-500">2초마다 자동 갱신 중...</span>
        )}
      </div>

      {data.status === "PUBLISHED" && <ProtectionSummary artwork={data} />}

      {data.status === "PUBLISHED" && (
        <div className="mb-6 flex items-center gap-3">
          <LikeButton artworkId={data.id} />
          <FollowButton creatorId={data.creatorId} />
          <ReportButton artworkId={data.id} />
        </div>
      )}

      {data.status === "FAILED" && data.errorMessage && (
        <p className="mb-6 rounded border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {data.errorMessage}
        </p>
      )}

      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
        <dt className="text-neutral-400">보호 프리셋</dt>
        <dd>{data.protectionProfile}</dd>
        <dt className="text-neutral-400">AI 학습 허용</dt>
        <dd>{data.allowAiTraining ? "허용" : "거부 (Do-Not-Train)"}</dd>
        {data.perceptualHash && (
          <>
            <dt className="text-neutral-400">perceptualHash</dt>
            <dd className="break-all font-mono text-xs">{data.perceptualHash}</dd>
          </>
        )}
      </dl>

      {record && (
        <div className="mt-6 rounded border border-neutral-800 px-4 py-3">
          <p className="mb-2 text-sm font-medium">온체인 소유권 등록</p>
          <a
            href={`https://amoy.polygonscan.com/tx/${record.txHash}`}
            target="_blank"
            rel="noreferrer"
            className="text-sm text-blue-400 underline"
          >
            Polygon Amoy 익스플로러에서 보기 ↗
          </a>
        </div>
      )}

      {data.status === "PUBLISHED" && <CommentSection artworkId={data.id} />}
    </div>
  );
}

function StatusBadge({ status }: { status: Artwork["status"] }) {
  const color =
    status === "PUBLISHED"
      ? "bg-green-900 text-green-300"
      : status === "FAILED"
        ? "bg-red-900 text-red-300"
        : "bg-neutral-800 text-neutral-300";
  return <span className={`rounded px-2 py-1 text-xs font-medium ${color}`}>{STATUS_LABEL[status]}</span>;
}
