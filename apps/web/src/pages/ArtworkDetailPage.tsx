import { useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../api/client";
import type { Artwork } from "../api/types";
import { getUser } from "../api/users";
import { ArtworkImage } from "../components/ArtworkImage";
import { Avatar } from "../components/Avatar";
import { LikeButton } from "../components/LikeButton";
import { FollowButton } from "../components/FollowButton";
import { CommentSection } from "../components/CommentSection";
import { ReportButton } from "../components/ReportButton";
import { ProtectionSummary } from "../components/ProtectionSummary";
import { useAuthStore } from "../store/auth";
import { useCoinBalance } from "../hooks/useCoinBalance";
import { setOriginalPreviewBlocked, unlockOriginalPreview } from "../api/originalPreview";
import { LoraGenerationPanel } from "../components/LoraGenerationPanel";
import { BotPolicyPanel } from "../components/BotPolicyPanel";

// coins.ts's COIN_COSTS.ORIGINAL_PREVIEW_UNLOCK -- display-only, see
// UploadPage.tsx's identical STRONG_PROTECTION_COIN_COST comment.
const ORIGINAL_PREVIEW_COIN_COST = 1;

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
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const user = useAuthStore((s) => s.user);
  const { refresh: refreshCoinBalance } = useCoinBalance();
  const [originalPreviewBusy, setOriginalPreviewBusy] = useState(false);
  const [originalPreviewError, setOriginalPreviewError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["artwork", id],
    queryFn: () => api.get<Artwork>(`/artworks/${id}`),
    enabled: Boolean(id),
    refetchInterval: (query) => (query.state.data && TERMINAL_STATUSES.includes(query.state.data.status) ? false : 2000),
  });

  const { data: creator } = useQuery({
    queryKey: ["user", data?.creatorId],
    queryFn: () => getUser(data!.creatorId),
    enabled: Boolean(data?.creatorId),
  });

  if (isLoading) return <p className="mt-12 text-center text-neutral-400">불러오는 중...</p>;
  if (error || !data) return <p className="mt-12 text-center text-red-400">작품을 찾을 수 없습니다.</p>;

  const record = data.ownershipRecords?.[0];
  const isCreator = user?.id === data.creatorId;

  async function refetchArtwork() {
    await queryClient.invalidateQueries({ queryKey: ["artwork", id] });
  }

  async function onToggleOriginalPreviewBlocked(blocked: boolean) {
    setOriginalPreviewBusy(true);
    setOriginalPreviewError(null);
    try {
      await setOriginalPreviewBlocked(data!.id, blocked);
      await refetchArtwork();
    } catch {
      setOriginalPreviewError("원본 미리보기 설정을 변경하지 못했습니다.");
    } finally {
      setOriginalPreviewBusy(false);
    }
  }

  async function onUnlockOriginalPreview() {
    setOriginalPreviewBusy(true);
    setOriginalPreviewError(null);
    try {
      await unlockOriginalPreview(data!.id);
      await Promise.all([refetchArtwork(), refreshCoinBalance()]);
    } catch (err) {
      if (err instanceof ApiError && err.status === 402) {
        const body = err.body as { required?: number; balance?: number } | undefined;
        setOriginalPreviewError(
          body?.required !== undefined && body?.balance !== undefined
            ? `코인이 부족해요. 원본 보기에는 코인 ${body.required}개가 필요한데, 현재 ${body.balance}개 갖고 계세요.`
            : "코인이 부족해요.",
        );
      } else {
        setOriginalPreviewError("원본 미리보기 열람에 실패했습니다.");
      }
    } finally {
      setOriginalPreviewBusy(false);
    }
  }

  // Cancel-upload feature (2026-08-14) -- same button/handler regardless of
  // status, per the requirement that it stay usable after the upload
  // finishes: asset-service's own POST /:id/cancel decides whether that
  // means "cancel the in-flight protect job" (still processing) or
  // "delete the artwork outright" (already published/failed) -- see that
  // route's own doc. Only the confirm copy and post-success navigation
  // differ here based on which case this is.
  async function onCancel() {
    const isTerminal = TERMINAL_STATUSES.includes(data!.status);
    const confirmed = window.confirm(
      isTerminal
        ? "이 작품을 삭제할까요? 되돌릴 수 없습니다."
        : "업로드를 취소할까요? 진행 중인 보호 처리가 중단됩니다.",
    );
    if (!confirmed) return;

    setCancelling(true);
    setCancelError(null);
    try {
      const result = await api.post<{ deleted?: boolean }>(`/artworks/${data!.id}/cancel`);
      if (result.deleted) {
        navigate("/my-artworks");
        return; // component unmounts on navigate -- no local state left to reset
      }
      await refetchArtwork();
    } catch {
      setCancelError("취소/삭제에 실패했습니다.");
    } finally {
      setCancelling(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl">
      <div className="mb-4 flex items-center justify-between gap-3">
        <Link to={`/creators/${data.creatorId}`} className="flex min-w-0 items-center gap-2.5">
          <Avatar seed={data.creatorId} label={creator?.displayName || creator?.handle || data.creatorId} avatarUri={creator?.avatarUri} size="sm" />
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-neutral-100">{creator?.displayName || creator?.handle || "..."}</p>
            <p className="truncate text-xs text-neutral-500">@{creator?.handle ?? data.creatorId}</p>
          </div>
        </Link>
        {isCreator && (
          <button
            type="button"
            disabled={cancelling}
            onClick={() => void onCancel()}
            className="shrink-0 rounded-full border border-red-900/60 px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-950/40 disabled:opacity-50"
          >
            {TERMINAL_STATUSES.includes(data.status) ? "작품 삭제" : "업로드 취소"}
          </button>
        )}
      </div>

      <ArtworkImage
        artworkId={data.id}
        hasVariants={data.assetVersions.length > 0}
        className="mb-4 w-full rounded-2xl border border-neutral-800 object-contain"
      />

      <h1 className="mb-1 text-xl font-bold text-neutral-50">{data.title}</h1>

      {data.tags.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-1.5">
          {data.tags.map((tag) => (
            <Link
              key={tag}
              to={`/feed?tag=${encodeURIComponent(tag)}`}
              className="rounded-full bg-neutral-800 px-2.5 py-1 text-xs text-neutral-300 transition-colors hover:bg-neutral-700 hover:text-neutral-100"
            >
              #{tag}
            </Link>
          ))}
        </div>
      )}

      <div className="mb-4 flex items-center gap-2">
        <StatusBadge status={data.status} />
        {!TERMINAL_STATUSES.includes(data.status) && (
          <span className="text-xs text-neutral-500">2초마다 자동 갱신 중...</span>
        )}
      </div>
      {cancelError && <p className="mb-4 text-xs text-red-400">{cancelError}</p>}

      {data.status === "PUBLISHED" && <ProtectionSummary artwork={data} />}

      {data.status === "PUBLISHED" && (
        <div className="mb-6 flex items-center gap-2.5">
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

      <dl className="card grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 p-4 text-sm">
        <dt className="text-neutral-500">보호 프리셋</dt>
        <dd className="text-neutral-200">{data.protectionProfile}</dd>
        <dt className="text-neutral-500">AI 학습 허용</dt>
        <dd className="text-neutral-200">{data.allowAiTraining ? "허용" : "거부 (Do-Not-Train)"}</dd>
        {data.perceptualHash && (
          <>
            <dt className="text-neutral-500">perceptualHash</dt>
            <dd className="break-all font-mono text-xs text-neutral-400">{data.perceptualHash}</dd>
          </>
        )}
      </dl>

      {record && (
        <div className="card mt-4 p-4">
          <p className="mb-2 text-sm font-medium text-neutral-200">온체인 소유권 등록</p>
          <a
            href={`https://amoy.polygonscan.com/tx/${record.txHash}`}
            target="_blank"
            rel="noreferrer"
            className="text-sm text-brand-400 underline"
          >
            Polygon Amoy 익스플로러에서 보기 ↗
          </a>
        </div>
      )}

      {data.status === "PUBLISHED" && (
        <div className="card mb-6 mt-4 p-4">
          {isCreator ? (
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={data.originalPreviewBlocked}
                disabled={originalPreviewBusy}
                onChange={(e) => void onToggleOriginalPreviewBlocked(e.target.checked)}
              />
              코인으로도 원본 공개 금지
            </label>
          ) : (
            !data.originalPreviewBlocked &&
            (data.originalPreviewUnlockedByViewer ? (
              <ArtworkImage
                artworkId={data.id}
                hasVariants
                variant="original"
                className="w-full rounded border border-neutral-800 object-contain"
              />
            ) : (
              <button
                type="button"
                disabled={originalPreviewBusy}
                onClick={() => void onUnlockOriginalPreview()}
                className="rounded bg-amber-800 px-3 py-2 text-sm font-medium text-amber-100 hover:bg-amber-700 disabled:opacity-50"
              >
                🪙 코인 {ORIGINAL_PREVIEW_COIN_COST}개로 원본 보기
              </button>
            ))
          )}
          {originalPreviewError && <p className="mt-2 text-xs text-red-400">{originalPreviewError}</p>}
        </div>
      )}

      {data.status === "PUBLISHED" && isCreator && <BotPolicyPanel artworkId={data.id} />}

      {data.status === "PUBLISHED" && isCreator && <LoraGenerationPanel artworkId={data.id} />}

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
