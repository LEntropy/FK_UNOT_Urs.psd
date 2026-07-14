import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";

const PRESETS = [
  { value: "L1_PREVIEW", label: "L1 · 미리보기 (약한 보호)" },
  { value: "L2_PORTFOLIO", label: "L2 · 포트폴리오 (중간 보호)" },
  { value: "L3_ANTI_TRAIN", label: "L3 · 학습 방지 우선 (강한 보호)" },
];

export function UploadPage() {
  const navigate = useNavigate();
  const [title, setTitle] = useState("");
  const [sourceImageUri, setSourceImageUri] = useState("");
  const [protectionProfile, setProtectionProfile] = useState("L3_ANTI_TRAIN");
  const [allowAiTraining, setAllowAiTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const res = await api.post<{ id: string }>("/artworks", {
        title,
        sourceImageUri,
        protectionProfile,
        allowAiTraining,
      });
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
      <p className="mb-6 text-sm text-neutral-400">
        아직 오브젝트 스토리지가 연동되지 않아, protection-svc/asset-service가 실제로 접근할 수 있는 서버 측
        로컬 경로를 입력해야 합니다 (PoC 범위 한계, apps/asset-service/README.md 참고).
      </p>
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
          이미지 경로
          <input
            required
            placeholder="C:/path/to/image.png"
            value={sourceImageUri}
            onChange={(e) => setSourceImageUri(e.target.value)}
            className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2"
          />
        </label>
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
