import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";

// coins.ts's COIN_COSTS.STRONG_PROTECTION on the asset-service side --
// display-only here (the server does the real check/charge), kept as a
// literal rather than fetched since this project's coin costs aren't
// user-configurable yet.
const STRONG_PROTECTION_COIN_COST = 1;

const PRESETS = [
  { value: "L1_PREVIEW", label: "L1 · 미리보기 (약한 보호)" },
  { value: "L2_PORTFOLIO", label: "L2 · 포트폴리오 (중간 보호)" },
  { value: "L3_ANTI_TRAIN", label: "L3 · 학습 방지 우선 (강한 보호)" },
  // 실제 LoRA 재학습 테스트로 검증된 유일한 등급(SD1.5+SDXL 순차 공격,
  // PHASE4_SCOPING.md §6) -- 다른 세 등급보다 훨씬 오래 걸림(수 분, 실제
  // GPU 작업)이라는 걸 라벨에서부터 명시.
  { value: "STRONG_PROTECTION", label: "강력 보호 · 실제 검증됨 (처리에 수 분 소요)" },
];

interface SuggestedTag {
  tag: string;
  score: number;
}

// Advanced-options upload feature (2026-08-08, user request) -- STRONG_
// PROTECTION only. hybrid_protect.py's own HYBRID_FULL preset (latent
// 0.15 + pixel top-up 0.02) is "권장" here, unchanged from the server-
// side default -- picking it sends no override at all (undefined), so a
// user who never opens this section gets exactly what they always did.
// "약하게"/"강하게" are reasoned points on the same tradeoff curve
// (smaller epsilon = closer to the original, less-tested resistance;
// larger = the opposite), not independently validated values -- see the
// warning copy below and hybrid_protect.py's own module doc for the real
// numbers this is based on (epsilon=0.3 was this project's own original,
// pre-recalibration value; epsilon=0.08 is a reasoned weaker point, not
// separately measured).
const EPSILON_PRESETS = [
  {
    key: "recommended" as const,
    label: "권장 (기본값)",
    latent: 0.15,
    pixel: 0.02,
    description: "이 프로젝트가 실측으로 확인한 균형점이에요. 대부분의 경우 이 값을 그대로 쓰는 걸 추천해요.",
  },
  {
    key: "weaker" as const,
    label: "약하게 (원본에 더 가깝게)",
    latent: 0.08,
    pixel: 0.01,
    description:
      "원본과 더 비슷해 보이도록 왜곡을 줄여요. 대신 AI 학습을 얼마나 잘 방해하는지는 권장값만큼 확인되지 않았어요.",
  },
  {
    key: "stronger" as const,
    label: "강하게 (방어 강화, 왜곡 커짐)",
    latent: 0.3,
    pixel: 0.02,
    description:
      "더 강하게 방해하려는 값이에요. 색 번짐이나 무늬가 두드러져서 원본을 알아보기 어려울 수 있고, 권장값보다 검증이 부족해요.",
  },
];

export function UploadPage() {
  const navigate = useNavigate();
  const [title, setTitle] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [protectionProfile, setProtectionProfile] = useState("L3_ANTI_TRAIN");
  // null = "권장" (no override sent, server-side default applies) --
  // deliberately not defaulted to EPSILON_PRESETS[0]'s own values here,
  // so a user who never touches this section changes nothing about what
  // gets sent, matching this feature's opt-in-only design.
  const [epsilonPresetKey, setEpsilonPresetKey] = useState<(typeof EPSILON_PRESETS)[number]["key"] | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [allowAiTraining, setAllowAiTraining] = useState(false);
  const [tags, setTags] = useState<string[]>([]);
  const [customTag, setCustomTag] = useState("");
  const [suggestingTags, setSuggestingTags] = useState(false);
  const [tagError, setTagError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Upload-preview step (PixAI-style image-to-tag): as soon as a file is
  // picked, ask protection-svc (via asset-service/api-gateway's
  // suggest-tags proxy) for a ranked tag guess -- shown to the user to
  // edit/remove/add to before the real upload, not applied silently.
  useEffect(() => {
    if (!file) {
      setTags([]);
      setTagError(null);
      return;
    }
    let cancelled = false;
    setSuggestingTags(true);
    setTagError(null);
    (async () => {
      try {
        const form = new FormData();
        form.set("image", file);
        const res = await api.upload<{ tags: SuggestedTag[] }>("/artworks/suggest-tags", form);
        if (!cancelled) setTags(res.tags.map((t) => t.tag));
      } catch {
        // Best-effort preview -- a failed suggestion shouldn't block picking
        // a file or block the real upload; the user can still type tags by hand.
        if (!cancelled) setTagError("태그 추천을 가져오지 못했습니다. 직접 입력해주세요.");
      } finally {
        if (!cancelled) setSuggestingTags(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [file]);

  function removeTag(tag: string) {
    setTags((prev) => prev.filter((t) => t !== tag));
  }

  function addCustomTag() {
    const trimmed = customTag.trim();
    if (trimmed && !tags.includes(trimmed)) {
      setTags((prev) => [...prev, trimmed]);
    }
    setCustomTag("");
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!file) {
      setError("이미지 파일을 선택해주세요.");
      return;
    }
    setSubmitting(true);
    try {
      const form = new FormData();
      form.set("title", title);
      form.set("protectionProfile", protectionProfile);
      // Only sent when STRONG_PROTECTION and the user picked something
      // other than "권장" -- otherwise no override field goes at all, so
      // the server-side default (hybrid_protect.py's own HYBRID_FULL)
      // applies exactly as it did before this feature existed.
      if (protectionProfile === "STRONG_PROTECTION" && epsilonPresetKey && epsilonPresetKey !== "recommended") {
        const preset = EPSILON_PRESETS.find((p) => p.key === epsilonPresetKey)!;
        form.set("strongProtectionLatentEpsilon", String(preset.latent));
        form.set("strongProtectionPixelEpsilon", String(preset.pixel));
      }
      form.set("allowAiTraining", String(allowAiTraining));
      form.set("tags", JSON.stringify(tags));
      form.set("image", file);

      const res = await api.upload<{ id: string }>("/artworks", form);
      navigate(`/artworks/${res.id}`);
    } catch (err) {
      if (err instanceof ApiError && err.status === 402) {
        const body = err.body as { required?: number; balance?: number } | undefined;
        setError(
          body?.required !== undefined && body?.balance !== undefined
            ? `코인이 부족해요. 강력 보호에는 코인 ${body.required}개가 필요한데, 현재 ${body.balance}개 갖고 계세요.`
            : "코인이 부족해요.",
        );
      } else {
        setError(err instanceof ApiError ? JSON.stringify(err.body) : "업로드에 실패했습니다.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto mt-12 max-w-lg">
      <h1 className="mb-2 text-2xl font-semibold">작품 업로드</h1>
      <form onSubmit={onSubmit} className="flex flex-col gap-4">
        <label className="flex flex-col gap-1 text-sm">
          제목
          <input
            required
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          이미지 파일
          <input
            required
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2"
          />
        </label>
        <div className="flex flex-col gap-1 text-sm">
          <span>태그{suggestingTags && <span className="ml-2 text-neutral-500">추천 태그 분석 중...</span>}</span>
          {tagError && <p className="text-xs text-amber-400">{tagError}</p>}
          <div className="flex flex-wrap gap-2">
            {tags.map((tag) => (
              <span
                key={tag}
                className="flex items-center gap-1 rounded-full bg-neutral-800 px-3 py-1 text-xs text-neutral-100"
              >
                {tag}
                <button
                  type="button"
                  onClick={() => removeTag(tag)}
                  aria-label={`${tag} 태그 삭제`}
                  className="text-neutral-400 hover:text-neutral-100"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              value={customTag}
              onChange={(e) => setCustomTag(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  addCustomTag();
                }
              }}
              placeholder="태그 직접 추가"
              className="flex-1 rounded border border-neutral-700 bg-neutral-900 px-3 py-2"
            />
            <button
              type="button"
              onClick={addCustomTag}
              className="rounded border border-neutral-700 px-3 py-2 text-sm text-neutral-100"
            >
              추가
            </button>
          </div>
        </div>
        <label className="flex flex-col gap-1 text-sm">
          보호 강도
          <select
            value={protectionProfile}
            onChange={(e) => {
              setProtectionProfile(e.target.value);
              if (e.target.value !== "STRONG_PROTECTION") {
                setEpsilonPresetKey(null);
                setAdvancedOpen(false);
              }
            }}
            className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2"
          >
            {PRESETS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
          {protectionProfile === "STRONG_PROTECTION" && (
            <>
              <p className="text-xs text-amber-300">🪙 코인 {STRONG_PROTECTION_COIN_COST}개가 소모돼요.</p>
              <p className="text-xs text-yellow-400">
                실제 GPU에서 LoRA 공격을 두 번(SD1.5, SDXL) 순차로 돌리는 방식이라 업로드 처리에 10분 이상 걸릴 수 있어요.
                일시적으로 실패하면 자동으로 L3 단계로 대체 처리됩니다.
                이 단계는 두 공격이 이어지며 효과가 서로 증폭되도록 검증된 방식이라, 완성된 이미지에 눈에 띄는 색
                번짐이나 왜곡이 나타날 수 있어요 — 다른 단계처럼 원본과 거의 구분되지 않는 수준이 아닙니다.
              </p>
              <div className="mt-2 rounded border border-neutral-800 bg-neutral-950/40 px-3 py-2">
                <button
                  type="button"
                  onClick={() => setAdvancedOpen((v) => !v)}
                  className="text-xs text-neutral-400 underline hover:text-neutral-200"
                >
                  {advancedOpen ? "고급 옵션 숨기기 ▲" : "고급 옵션 (원본과 얼마나 비슷하게 남길지 직접 조절) ▼"}
                </button>
                {advancedOpen && (
                  <div className="mt-3 flex flex-col gap-2">
                    <p className="text-xs text-neutral-500">
                      "원본과 얼마나 비슷하게 보일지"와 "AI 학습을 얼마나 잘 방해할지"는 서로 트레이드오프예요. 권장값이
                      아닌 다른 값은 이 프로젝트가 실제로 LoRA를 재학습시켜 검증한 것이 아니라서, 방어 효과가 권장값만큼
                      확실하지 않을 수 있어요.
                    </p>
                    <div className="flex flex-col gap-2">
                      {EPSILON_PRESETS.map((p) => {
                        const selected = (epsilonPresetKey ?? "recommended") === p.key;
                        return (
                          <button
                            key={p.key}
                            type="button"
                            onClick={() => setEpsilonPresetKey(p.key === "recommended" ? null : p.key)}
                            className={`rounded border px-3 py-2 text-left text-xs ${
                              selected
                                ? "border-neutral-100 bg-neutral-800"
                                : "border-neutral-800 bg-neutral-950/40 hover:border-neutral-700"
                            }`}
                          >
                            <span className="font-medium text-neutral-100">{p.label}</span>
                            <p className="mt-1 text-neutral-400">{p.description}</p>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={allowAiTraining} onChange={(e) => setAllowAiTraining(e.target.checked)} />
          AI 학습 허용
        </label>
        {error && <p className="text-sm text-red-400">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="rounded bg-neutral-100 px-3 py-2 font-medium text-neutral-900 disabled:opacity-50"
        >
          {submitting ? "업로드 중..." : "업로드"}
        </button>
      </form>
    </div>
  );
}
