import { useState } from "react";
import type { Artwork } from "../api/types";

const TIER_LABEL: Record<Artwork["protectionProfile"], string> = {
  L1_PREVIEW: "미리보기 보호",
  L2_PORTFOLIO: "포트폴리오 보호",
  L3_ANTI_TRAIN: "AI 학습 방지 강화",
};

const TIER_DESCRIPTION: Record<Artwork["protectionProfile"], string> = {
  L1_PREVIEW: "가볍게 처리해 빠르게 미리 볼 수 있는 단계예요.",
  L2_PORTFOLIO: "포트폴리오 공개에 맞춘 균형 잡힌 보호 단계예요.",
  L3_ANTI_TRAIN: "AI 학습을 방해하는 데 가장 강하게 처리한 단계예요.",
};

const TIER_STRENGTH: Record<Artwork["protectionProfile"], number> = {
  L1_PREVIEW: 1,
  L2_PORTFOLIO: 2,
  L3_ANTI_TRAIN: 3,
};

/**
 * Non-technical-facing summary of what protection was actually applied --
 * asked for explicitly: something a layperson can read at a glance instead
 * of raw technical fields. Deliberately does not turn the real measured
 * numbers (styleDriftScore etc, protection-svc's evaluate.py) into a fake
 * "94% protected" style percentage -- there's no real 0-100% scale a Gram-
 * matrix cosine drift maps onto, and inventing one would be exactly the
 * kind of overclaim this project's other docs (PHASE4_SCOPING.md,
 * ml-engine/README.md) have been careful to avoid. Instead: a strength
 * meter tied to the real, deterministic preset tier the user/system chose
 * (always true, no measurement needed), plus a checklist of concrete,
 * true-or-false facts, plus an optional "measured" confirmation line only
 * shown when a real number backs it up.
 */
export function ProtectionSummary({ artwork }: { artwork: Artwork }) {
  const [showDetails, setShowDetails] = useState(false);

  const strength = TIER_STRENGTH[artwork.protectionProfile];
  const hasOwnershipRecord = artwork.ownershipRecords.length > 0;
  const hasMeasuredEffect = typeof artwork.styleDriftScore === "number" && artwork.styleDriftScore > 0;

  return (
    <div className="mb-6 rounded border border-neutral-800 bg-neutral-950/40 px-4 py-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-sm font-medium">이 그림은 이렇게 보호됐어요</span>
        <div className="flex items-center gap-1" aria-hidden="true">
          {[1, 2, 3].map((n) => (
            <span
              key={n}
              className={`h-2 w-6 rounded-full ${n <= strength ? "bg-green-500" : "bg-neutral-800"}`}
            />
          ))}
        </div>
      </div>

      <p className="mb-3 text-sm">
        <span className="font-medium text-green-400">{TIER_LABEL[artwork.protectionProfile]}</span>
        <span className="text-neutral-400"> · {TIER_DESCRIPTION[artwork.protectionProfile]}</span>
      </p>

      <ul className="mb-3 space-y-1.5 text-sm">
        <ProtectionFact done>사람 눈에는 원본과 거의 똑같아 보이도록 처리했어요</ProtectionFact>
        <ProtectionFact done>AI가 그림을 오해하도록 픽셀을 미세하게 바꿨어요</ProtectionFact>
        <ProtectionFact done>보이지 않는 워터마크를 심어 나중에 무단 사용을 추적할 수 있어요</ProtectionFact>
        <ProtectionFact done={hasOwnershipRecord}>블록체인에 소유권을 등록해 제작 시점을 증명해요</ProtectionFact>
        {hasMeasuredEffect && (
          <ProtectionFact done>실제로 AI가 인식하는 특징이 달라진 것을 측정으로 확인했어요</ProtectionFact>
        )}
      </ul>

      <button
        onClick={() => setShowDetails((v) => !v)}
        className="text-xs text-neutral-500 underline hover:text-neutral-300"
      >
        {showDetails ? "자세한 수치 숨기기" : "자세한 수치 보기 (기술적인 내용)"}
      </button>

      {showDetails && (
        <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 border-t border-neutral-800 pt-3 text-xs">
          <dt className="text-neutral-500">보호 프리셋</dt>
          <dd className="text-neutral-300">{artwork.protectionProfile}</dd>
          <dt className="text-neutral-500">화풍 인식 변화도</dt>
          <dd className="text-neutral-300">
            {typeof artwork.styleDriftScore === "number" ? artwork.styleDriftScore.toFixed(4) : "측정 안 됨"}
          </dd>
          <dt className="text-neutral-500">원본과의 시각적 유사도 (PSNR)</dt>
          <dd className="text-neutral-300">
            {typeof artwork.perceptualPsnrDb === "number" ? `${artwork.perceptualPsnrDb.toFixed(1)} dB` : "측정 안 됨"}
          </dd>
        </dl>
      )}
    </div>
  );
}

function ProtectionFact({ done, children }: { done: boolean; children: React.ReactNode }) {
  return (
    <li className={`flex items-start gap-2 ${done ? "text-neutral-200" : "text-neutral-500"}`}>
      <span className={done ? "text-green-400" : "text-neutral-600"}>{done ? "✓" : "–"}</span>
      <span>{children}</span>
    </li>
  );
}
