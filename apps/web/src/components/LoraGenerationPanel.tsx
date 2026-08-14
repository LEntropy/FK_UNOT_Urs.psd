import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError } from "../api/client";
import { createLoraJob, getLoraJob, downloadLoraJob } from "../api/loraJobs";
import { useCoinBalance } from "../hooks/useCoinBalance";

// coins.ts's COIN_COSTS.LORA_GENERATION -- display-only, see
// UploadPage.tsx's identical STRONG_PROTECTION_COIN_COST comment.
const LORA_GENERATION_COIN_COST = 1;

const STATUS_LABEL: Record<string, string> = {
  QUEUED: "대기 중",
  RUNNING: "학습 중 (수 분 소요될 수 있어요)",
  COMPLETED: "완료",
  FAILED: "실패",
};

/** Creator-only: spend coins to train and download a real LoRA from this
 * artwork's own original image (protection-svc's ml-engine/src/
 * lora_generate.py -- single-image SD1.5 LoRA, coin-system feature). */
export function LoraGenerationPanel({ artworkId }: { artworkId: string }) {
  const { refresh: refreshCoinBalance } = useCoinBalance();
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const { data: job } = useQuery({
    queryKey: ["loraJob", jobId],
    queryFn: () => getLoraJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) =>
      query.state.data && (query.state.data.status === "COMPLETED" || query.state.data.status === "FAILED") ? false : 5000,
  });

  async function onSubmit() {
    setSubmitting(true);
    setError(null);
    try {
      const res = await createLoraJob(artworkId);
      setJobId(res.id);
      await refreshCoinBalance();
    } catch (err) {
      if (err instanceof ApiError && err.status === 402) {
        const body = err.body as { required?: number; balance?: number } | undefined;
        setError(
          body?.required !== undefined && body?.balance !== undefined
            ? `코인이 부족해요. LoRA 생성에는 코인 ${body.required}개가 필요한데, 현재 ${body.balance}개 갖고 계세요.`
            : "코인이 부족해요.",
        );
      } else {
        setError("LoRA 생성 요청에 실패했습니다.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  async function onDownload() {
    if (!jobId) return;
    try {
      await downloadLoraJob(jobId);
    } catch {
      setError("다운로드에 실패했습니다.");
    }
  }

  return (
    <div className="mb-6 rounded border border-neutral-800 px-4 py-3">
      <p className="mb-2 text-sm font-medium">이 그림으로 LoRA 만들기</p>
      <p className="mb-2 text-xs text-neutral-500">
        원본 이미지 한 장으로 실제 사용 가능한 SD1.5 LoRA 파일을 학습해서 다운로드해요. 이미지 한 장만 쓰는 방식이라
        여러 장을 쓴 LoRA보다 품질이 제한적일 수 있어요.
      </p>
      {!jobId && (
        <button
          type="button"
          disabled={submitting}
          onClick={() => void onSubmit()}
          className="rounded bg-amber-800 px-3 py-2 text-sm font-medium text-amber-100 hover:bg-amber-700 disabled:opacity-50"
        >
          🪙 코인 {LORA_GENERATION_COIN_COST}개로 LoRA 생성
        </button>
      )}
      {jobId && job && (
        <div className="flex items-center gap-3">
          <span className="text-sm text-neutral-300">{STATUS_LABEL[job.status] ?? job.status}</span>
          {job.status === "COMPLETED" && (
            <button
              type="button"
              onClick={() => void onDownload()}
              className="rounded bg-neutral-100 px-3 py-2 text-sm font-medium text-neutral-900 hover:bg-neutral-300"
            >
              LoRA 다운로드
            </button>
          )}
          {job.status === "FAILED" && job.errorMessage && <span className="text-xs text-red-400">{job.errorMessage}</span>}
        </div>
      )}
      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
    </div>
  );
}
