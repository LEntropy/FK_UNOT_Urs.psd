import { useState } from "react";
import type { Artwork } from "../api/types";

const TIER_LABEL: Record<Artwork["protectionProfile"], string> = {
  L1_PREVIEW: "미리보기 보호",
  L2_PORTFOLIO: "포트폴리오 보호",
  L3_ANTI_TRAIN: "AI 학습 방지 강화",
  STRONG_PROTECTION: "강력 보호 (실제 검증됨)",
};

const TIER_DESCRIPTION: Record<Artwork["protectionProfile"], string> = {
  L1_PREVIEW: "가볍게 처리해 빠르게 미리 볼 수 있는 단계예요.",
  L2_PORTFOLIO: "포트폴리오 공개에 맞춘 균형 잡힌 보호 단계예요.",
  L3_ANTI_TRAIN: "AI 학습을 방해하는 데 가장 강하게 처리한 단계예요.",
  STRONG_PROTECTION: "실제 LoRA 재학습 실험으로 방어 효과를 확인한, 가장 강력한 단계예요.",
};

const TIER_STRENGTH: Record<Artwork["protectionProfile"], number> = {
  L1_PREVIEW: 1,
  L2_PORTFOLIO: 2,
  L3_ANTI_TRAIN: 3,
  STRONG_PROTECTION: 4,
};

const MAX_TIER_STRENGTH = 4;

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

  // What was *requested* (protectionProfile) and what actually *ran*
  // (usedStrongProtection) can differ -- the dual-arch RunPod Serverless
  // attack falls back to plain style_cloak on any failure rather than
  // fail the whole upload (orchestrate.py's own strong_protection
  // branch). Showing "강력 보호" when that fallback silently happened
  // would be exactly the kind of overclaim this component's own doc
  // comment says to avoid -- fall the *display* tier back to
  // L3_ANTI_TRAIN too in that case, with an explicit note.
  const requestedStrongProtection = artwork.protectionProfile === "STRONG_PROTECTION";
  const fellBackFromStrongProtection = requestedStrongProtection && !artwork.usedStrongProtection;
  const displayTier: Artwork["protectionProfile"] = fellBackFromStrongProtection
    ? "L3_ANTI_TRAIN"
    : artwork.protectionProfile;

  const strength = TIER_STRENGTH[displayTier];
  const hasOwnershipRecord = artwork.ownershipRecords.length > 0;
  const hasMeasuredEffect = typeof artwork.styleDriftScore === "number" && artwork.styleDriftScore > 0;
  // STRONG_PROTECTION's SD1.5->SDXL chained attack was validated at n=30
  // to *amplify* the SD1.5-stage effect (~4-5x single-stage, see
  // remote_dual_arch_cloak()'s docstring / [[lora-protection-research]]) --
  // that's the mechanism's actual protective value, but it also means the
  // output visibly departs from the original, unlike the subtle L1-L3
  // perturbations. Showing the same "looks nearly identical" claim here
  // would overclaim on a tier that, when it actually ran, looks visibly
  // different -- so this tier gets its own honest fact instead.
  const isVisiblyStrong = displayTier === "STRONG_PROTECTION";

  return (
    <div className="mb-6 rounded border border-neutral-800 bg-neutral-950/40 px-4 py-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-sm font-medium">이 그림은 이렇게 보호됐어요</span>
        <div className="flex items-center gap-1" aria-hidden="true">
          {Array.from({ length: MAX_TIER_STRENGTH }, (_, i) => i + 1).map((n) => (
            <span
              key={n}
              className={`h-2 w-6 rounded-full ${n <= strength ? "bg-green-500" : "bg-neutral-800"}`}
            />
          ))}
        </div>
      </div>

      <p className="mb-3 text-sm">
        <span className="font-medium text-green-400">{TIER_LABEL[displayTier]}</span>
        <span className="text-neutral-400"> · {TIER_DESCRIPTION[displayTier]}</span>
      </p>

      {fellBackFromStrongProtection && (
        <p className="mb-3 rounded border border-yellow-900 bg-yellow-950/20 px-3 py-2 text-xs text-yellow-300">
          강력 보호를 요청했지만 일시적인 문제로 적용하지 못해, 대신 AI 학습 방지 강화 단계로 처리됐어요.
        </p>
      )}

      <ul className="mb-3 space-y-1.5 text-sm">
        {isVisiblyStrong ? (
          <ProtectionFact done>
            방어 효과를 극대화하는 방식이라 원본과 눈에 띄게 달라 보일 수 있어요 (색 번짐/왜곡)
          </ProtectionFact>
        ) : (
          <ProtectionFact done>사람 눈에는 원본과 거의 똑같아 보이도록 처리했어요</ProtectionFact>
        )}
        <ProtectionFact done>AI가 그림을 오해하도록 픽셀을 미세하게 바꿨어요</ProtectionFact>
        <ProtectionFact done>보이지 않는 워터마크를 심어 나중에 무단 사용을 추적할 수 있어요</ProtectionFact>
        <ProtectionFact done={hasOwnershipRecord}>블록체인에 소유권을 등록해 제작 시점을 증명해요</ProtectionFact>
        {hasMeasuredEffect && (
          <ProtectionFact done>실제로 AI가 인식하는 특징이 달라진 것을 측정으로 확인했어요</ProtectionFact>
        )}
        {artwork.usedStrongProtection && (
          <ProtectionFact done>
            실제로 LoRA를 재학습시켜 방어 효과가 통계적으로 확인된 방식이 적용됐어요 (SD1.5 + SDXL 모두)
          </ProtectionFact>
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
          {requestedStrongProtection && (
            <>
              <dt className="text-neutral-500">강력 보호 실제 적용 여부</dt>
              <dd className="text-neutral-300">{artwork.usedStrongProtection ? "적용됨" : "적용 안 됨 (대체 처리)"}</dd>
            </>
          )}
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
