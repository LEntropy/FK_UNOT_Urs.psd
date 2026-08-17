import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";

// coins.ts's COIN_COSTS.STRONG_PROTECTION on the asset-service side --
// display-only here (the server does the real check/charge), kept as a
// literal rather than fetched since this project's coin costs aren't
// user-configurable yet.
const STRONG_PROTECTION_COIN_COST = 1;

// "강도" (intensity) advanced option (2026-08-15) -- see clean_protect.py's
// EPSILON_OVERRIDE_MIN/MAX and _scale_preset doc for where these numbers
// come from. RECOMMENDED matches CLEAN_FULL's own sd15_epsilon exactly, so
// leaving the slider untouched and submitting it would be a no-op -- the
// checkbox below still gates whether it's sent at all, so a user who never
// opens this section changes nothing about what gets sent (same pattern
// the old two-epsilon version of this feature used).
const STRONG_PROTECTION_EPSILON_MIN = 0.01;
const STRONG_PROTECTION_EPSILON_MAX = 0.3;
const STRONG_PROTECTION_EPSILON_RECOMMENDED = 0.05;

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

export function UploadPage() {
  const navigate = useNavigate();
  const [title, setTitle] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [protectionProfile, setProtectionProfile] = useState("L3_ANTI_TRAIN");
  const [customEpsilonEnabled, setCustomEpsilonEnabled] = useState(false);
  const [strongProtectionEpsilon, setStrongProtectionEpsilon] = useState(STRONG_PROTECTION_EPSILON_RECOMMENDED);
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
      if (protectionProfile === "STRONG_PROTECTION" && customEpsilonEnabled) {
        form.set("strongProtectionEpsilon", String(strongProtectionEpsilon));
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
                setCustomEpsilonEnabled(false);
                setStrongProtectionEpsilon(STRONG_PROTECTION_EPSILON_RECOMMENDED);
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
                실제 GPU에서 LoRA 공격을 SD1.5·SDXL 각각 2단계(총 4단계)로 돌리는 방식이라 업로드 처리에 수십 분
                걸릴 수 있어요. 일시적으로 실패하면 자동으로 L3 단계로 대체 처리됩니다. 색 번짐은 없지만, 하늘처럼
                넓고 평평한 영역엔 옅은 무늬가 남을 수 있어요 — 다른 단계처럼 원본과 완전히 구분되지 않는 수준은
                아닙니다.
              </p>
              <div className="mt-2 rounded border border-neutral-800 bg-neutral-950/40 px-3 py-3">
                <label className="flex items-center gap-2 text-xs text-neutral-300">
                  <input
                    type="checkbox"
                    checked={customEpsilonEnabled}
                    onChange={(e) => setCustomEpsilonEnabled(e.target.checked)}
                  />
                  강도 직접 설정
                </label>
                {!customEpsilonEnabled ? (
                  <p className="mt-1 text-xs text-neutral-500">
                    기본값을 쓰면 이 프로젝트가 실제로 측정해 확인한 균형점(epsilon {STRONG_PROTECTION_EPSILON_RECOMMENDED})으로
                    처리돼요. 잘 모르겠으면 건드리지 않는 걸 추천해요.
                  </p>
                ) : (
                  <div className="mt-3 flex flex-col gap-2">
                    <p className="text-xs text-neutral-500">
                      <strong className="text-neutral-300">epsilon</strong>은 그림에 섞는, 눈에는 잘 안 보이는 방해
                      신호의 세기예요. 값이 클수록 AI 학습을 더 강하게 방해하지만, 그만큼 그림 자체의 색·질감도 더
                      많이 바뀔 수 있어요 — "더 강하게 = 항상 더 좋게"가 아니라 방어력과 원본 보존 사이의
                      트레이드오프예요. {STRONG_PROTECTION_EPSILON_RECOMMENDED}가 이 프로젝트가 실제로 측정해본
                      기준값이고, 그보다 큰 값은 아직 실측하지 않은 구간이라 방어 효과가 항상 더 세진다고 보장되지
                      않아요.
                    </p>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min={STRONG_PROTECTION_EPSILON_MIN}
                        max={STRONG_PROTECTION_EPSILON_MAX}
                        step={0.01}
                        value={strongProtectionEpsilon}
                        onChange={(e) => setStrongProtectionEpsilon(Number(e.target.value))}
                        className="flex-1"
                      />
                      <span className="w-12 shrink-0 text-right font-mono text-xs text-neutral-200">
                        {strongProtectionEpsilon.toFixed(2)}
                      </span>
                    </div>
                    <p className="text-xs text-neutral-500">
                      {strongProtectionEpsilon <= STRONG_PROTECTION_EPSILON_RECOMMENDED
                        ? "약하게 — 원본에 더 가깝지만, 이 값에서의 방어 효과는 따로 검증되지 않았어요."
                        : strongProtectionEpsilon <= STRONG_PROTECTION_EPSILON_RECOMMENDED * 2
                          ? "권장값보다 강하게 — 방어력이 더 세질 가능성이 있지만 아직 실측된 값은 아니에요."
                          : "많이 강하게 — 하늘처럼 넓고 평평한 영역에 무늬가 더 뚜렷하게 보일 수 있어요."}
                    </p>
                    {strongProtectionEpsilon > STRONG_PROTECTION_EPSILON_RECOMMENDED * 4 && (
                      <p className="text-xs text-red-400">
                        ⚠ {STRONG_PROTECTION_EPSILON_RECOMMENDED * 4} 이상은 이 프로젝트의 과거 실험에서 원본을
                        알아보기 어려울 정도로 왜곡됐던 값에 가까워요. 신중하게 선택해주세요.
                      </p>
                    )}
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
