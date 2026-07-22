import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";

const PRESETS = [
  { value: "L1_PREVIEW", label: "L1 · 미리보기 (약한 보호)" },
  { value: "L2_PORTFOLIO", label: "L2 · 포트폴리오 (중간 보호)" },
  { value: "L3_ANTI_TRAIN", label: "L3 · 학습 방지 우선 (강한 보호)" },
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
      form.set("allowAiTraining", String(allowAiTraining));
      form.set("tags", JSON.stringify(tags));
      form.set("image", file);

      const res = await api.upload<{ id: string }>("/artworks", form);
      navigate(`/artworks/${res.id}`);
    } catch (err) {
      setError(err instanceof ApiError ? JSON.stringify(err.body) : "업로드에 실패했습니다.");
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
            onChange={(e) => setProtectionProfile(e.target.value)}
            className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2"
          >
            {PRESETS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={allowAiTraining} onChange={(e) => setAllowAiTraining(e.target.checked)} />
          AI 학습 허용 (온체인 doNotTrain 플래그가 반대로 기록됩니다)
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
