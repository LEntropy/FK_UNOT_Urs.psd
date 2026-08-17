import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import * as delivery from "../api/delivery";
import * as detection from "../api/detection";
import * as protection from "../api/protection";
import type { RemeasureResult } from "../api/protection";
import type { Artwork, DetectionCase, EvidenceBundle } from "../api/types";
import { ArtworkImage } from "../components/ArtworkImage";

// Same real-effect threshold this project's own internal validation script
// uses for styleDriftScore (apps/protection-svc/ml-engine/src/evaluate.py's
// verdict_style: "PASS" if drift > 0.05 else "WEAK/FAIL") -- the "테스트"
// tab's protection test reuses this exact number rather than inventing a
// separate one, so a user sees the same call our own validation would make.
const STYLE_DRIFT_PASS_THRESHOLD = 0.05;

type Tab = "protection" | "detection";

export function TestLabPage() {
  const [tab, setTab] = useState<Tab>("protection");
  const [artworkId, setArtworkId] = useState<string | null>(null);

  const { data: artworks, isLoading } = useQuery({
    queryKey: ["artworks"],
    queryFn: () => api.get<Artwork[]>("/artworks"),
  });
  const publishedArtworks = artworks?.filter((a) => a.status === "PUBLISHED") ?? [];
  const selected = publishedArtworks.find((a) => a.id === artworkId) ?? null;

  return (
    <div className="mx-auto mt-8 max-w-2xl">
      <h1 className="mb-1 text-2xl font-semibold">테스트 랩</h1>
      <p className="mb-6 text-sm text-neutral-400">
        내 작품에 실제로 적용된 보호 효과를, 우리 팀이 내부적으로 검증할 때 쓰는 방법을 그대로 적용해 직접 확인해보세요.
      </p>

      <div className="mb-6 flex gap-2 border-b border-neutral-800">
        <TabButton active={tab === "protection"} onClick={() => setTab("protection")}>
          보호 강도 테스트
        </TabButton>
        <TabButton active={tab === "detection"} onClick={() => setTab("detection")}>
          추적·증빙 테스트
        </TabButton>
      </div>

      <label className="mb-6 flex flex-col gap-1 text-sm">
        테스트할 작품
        <select
          value={artworkId ?? ""}
          onChange={(e) => setArtworkId(e.target.value || null)}
          className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2"
        >
          <option value="">작품을 선택하세요</option>
          {publishedArtworks.map((a) => (
            <option key={a.id} value={a.id}>
              {a.title}
            </option>
          ))}
        </select>
      </label>

      {isLoading && <p className="text-sm text-neutral-400">불러오는 중...</p>}
      {!isLoading && publishedArtworks.length === 0 && (
        <p className="text-sm text-neutral-400">
          공개된(보호 처리가 끝난) 작품이 없습니다. 먼저 작품을 업로드하고 보호 처리가 끝날 때까지 기다려주세요.
        </p>
      )}

      {selected && (
        <>
          <ArtworkImage
            artworkId={selected.id}
            hasVariants={selected.assetVersions.length > 0}
            className="mb-6 max-h-80 w-full rounded border border-neutral-800 object-contain"
          />
          {tab === "protection" ? (
            <>
              <ProtectionTest artwork={selected} />
              {selected.usedStrongProtection && <RealLoraScoreTest artwork={selected} />}
            </>
          ) : (
            <DetectionTest artwork={selected} />
          )}
        </>
      )}
    </div>
  );
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`border-b-2 px-3 py-2 text-sm font-medium ${
        active ? "border-neutral-100 text-neutral-100" : "border-transparent text-neutral-500 hover:text-neutral-300"
      }`}
    >
      {children}
    </button>
  );
}

function ProtectionTest({ artwork }: { artwork: Artwork }) {
  const hasMeasurement = typeof artwork.styleDriftScore === "number";

  const remeasure = useMutation({
    mutationFn: () => protection.remeasureProtection(artwork.id),
  });

  return (
    <div className="rounded border border-neutral-800 bg-neutral-950/40 px-4 py-4">
      <p className="mb-4 text-sm text-neutral-400">
        업로드 시점에 protection-svc가 이 작품에 대해 실제로 측정한 값이에요. AI가 그림을 "본다"고 할 때 실제로
        참고하는 특징 표현(VGG19 스타일 특징)을 원본과 보호본에서 각각 뽑아 비교하는 방식으로, 우리 팀의 LoRA 학습
        재현 검증에서도 같은 계열의 측정을 사용합니다.
      </p>

      {!hasMeasurement && (
        <p className="mb-4 rounded border border-neutral-800 bg-neutral-900 px-3 py-3 text-sm text-neutral-400">
          이 작품은 보호 효과 측정값이 저장되어 있지 않습니다 (측정이 스킵되었거나 이전 버전으로 처리된 작품일 수
          있어요).
        </p>
      )}

      {hasMeasurement && (
        <MeasurementCard
          title="업로드 시 측정값"
          styleDriftScore={artwork.styleDriftScore as number}
          perceptualPsnrDb={artwork.perceptualPsnrDb}
          styleSimilarityToOriginal={artwork.styleSimilarityToOriginal}
        />
      )}

      <div className="mt-4 border-t border-neutral-800 pt-4">
        <button
          onClick={() => remeasure.mutate()}
          disabled={remeasure.isPending}
          className="rounded bg-neutral-100 px-3 py-2 text-sm font-medium text-neutral-900 disabled:opacity-50"
        >
          {remeasure.isPending ? "재측정 중... (GPU에 위임, 몇 초 걸려요)" : "지금 다시 실행"}
        </button>
        <p className="mt-1 text-xs text-neutral-500">
          저장된 값을 덮어쓰지 않고, 원본을 그때그때 복호화해서 실제로 지금 배포 중인 이미지로 다시 측정해요.
        </p>

        {remeasure.isError && (
          <p className="mt-3 text-sm text-red-400">
            재측정에 실패했습니다 ({remeasure.error instanceof Error ? remeasure.error.message : "알 수 없는 오류"}).
          </p>
        )}

        {remeasure.isSuccess && (
          <div className="mt-3">
            <MeasurementCard
              title="방금 재실행한 결과 (라이브)"
              styleDriftScore={remeasure.data.styleDriftScore}
              perceptualPsnrDb={remeasure.data.perceptualPsnrDb}
              styleSimilarityToOriginal={remeasure.data.styleSimilarityToOriginal}
              accent
            />
          </div>
        )}
      </div>

      <p className="mt-4 text-xs text-neutral-500">
        참고: 이 수치는 실제 LoRA 재학습 없이 스타일 특징 거리만 비교한 값이에요. 우리 팀이 실제 GPU로 LoRA를
        재학습시켜 검증한 결과에서는, 이 효과가 이미지에 따라 약하게 나타나기도 한다는 점을 확인했습니다 (완벽한
        차단을 보장하지 않음).
      </p>
    </div>
  );
}

function MeasurementCard({
  title,
  styleDriftScore,
  perceptualPsnrDb,
  styleSimilarityToOriginal,
  accent = false,
}: {
  title: string;
  styleDriftScore: RemeasureResult["styleDriftScore"];
  perceptualPsnrDb: RemeasureResult["perceptualPsnrDb"];
  styleSimilarityToOriginal: RemeasureResult["styleSimilarityToOriginal"];
  accent?: boolean;
}) {
  const passes = typeof styleDriftScore === "number" && styleDriftScore > STYLE_DRIFT_PASS_THRESHOLD;

  return (
    <div className={`rounded border px-3 py-3 ${accent ? "border-blue-900 bg-blue-950/20" : "border-neutral-800"}`}>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-medium text-neutral-400">{title}</span>
        {typeof styleDriftScore === "number" && (
          <span
            className={`rounded px-2 py-0.5 text-xs font-medium ${
              passes ? "bg-green-900/40 text-green-300" : "bg-amber-900/40 text-amber-300"
            }`}
          >
            {passes ? "✓ 유의미한 변화" : "△ 약한 변화"}
          </span>
        )}
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
        <dt className="text-neutral-400">화풍 인식 변화도 (styleDriftScore)</dt>
        <dd className="font-mono">{typeof styleDriftScore === "number" ? styleDriftScore.toFixed(4) : "측정 안 됨"}</dd>
        <dt className="text-neutral-400">원본과의 시각적 유사도 (PSNR)</dt>
        <dd className="font-mono">{typeof perceptualPsnrDb === "number" ? `${perceptualPsnrDb.toFixed(1)} dB` : "측정 안 됨"}</dd>
        <dt className="text-neutral-400">원본 화풍과의 유사도</dt>
        <dd className="font-mono">
          {typeof styleSimilarityToOriginal === "number" ? styleSimilarityToOriginal.toFixed(4) : "측정 안 됨"}
        </dd>
      </dl>
    </div>
  );
}

const SCORE_VERDICT_LABEL: Record<protection.ScoreProtectionArchResult["verdict"], string> = {
  PROTECTED: "보호됨",
  WEAK: "약한 효과",
  NOT_PROTECTED: "효과 없음",
};

/**
 * Test Lab's on-demand real-LoRA-training protection score -- only rendered
 * for artworks strong_protection actually ran on (see this file's own
 * TestLabPage gating above). Unlike ProtectionTest's remeasure button (a
 * fast proxy-metric re-run), this triggers a real several-minutes RunPod
 * Serverless job that trains four LoRAs (SD1.5/SDXL x baseline/protected)
 * and shows the actual generated samples side by side -- the same kind of
 * evidence this project's own n=30 dual_arch_validation used internally,
 * just run once against this specific artwork instead of n=30 images for a
 * statistical claim (see protection_score.py's module doc).
 */
function RealLoraScoreTest({ artwork }: { artwork: Artwork }) {
  const [jobId, setJobId] = useState<string | null>(null);

  // The persisted last real result -- fetched on mount so a user who
  // closed the tab mid-run (or just comes back later) still sees it
  // without needing to remember a jobId that only ever lived in this
  // component's own state. Skipped once a live job is being tracked
  // (jobQuery below takes over as the source of truth then).
  const storedQuery = useQuery({
    queryKey: ["storedScoreProtectionResult", artwork.id],
    queryFn: () => protection.getStoredScoreProtectionResult(artwork.id),
    enabled: !jobId,
  });

  const start = useMutation({
    mutationFn: () => protection.createScoreProtectionJob(artwork.id),
    onSuccess: (res) => setJobId(res.jobId),
  });

  const jobQuery = useQuery({
    queryKey: ["scoreProtectionJob", artwork.id, jobId],
    queryFn: () => protection.getScoreProtectionJob(artwork.id, jobId!),
    enabled: Boolean(jobId),
    refetchInterval: (query) =>
      query.state.data && (query.state.data.status === "completed" || query.state.data.status === "failed")
        ? false
        : 5000,
    // Same reasoning as DetectionTest's caseQuery above -- this job takes
    // minutes, a user tabbing away shouldn't freeze the progress display.
    refetchIntervalInBackground: true,
  });

  // Live job (this tab started it) takes priority; otherwise fall back to
  // whatever was last actually persisted for this artwork, from this tab
  // or any earlier one.
  const job = jobQuery.data ?? storedQuery.data ?? undefined;
  const isLiveJob = Boolean(jobId);

  return (
    <div className="mt-6 rounded border border-neutral-800 bg-neutral-950/40 px-4 py-4">
      <p className="mb-1 text-sm font-medium">실제 LoRA 재학습 테스트</p>
      <p className="mb-4 text-sm text-neutral-400">
        이 작품은 강력 보호(실제 GPU에서 LoRA를 학습시켜 검증하는 방식)가 실제로 적용됐어요. 같은 방식으로 지금 직접
        재학습을 돌려서, 보호되지 않은 원본으로 학습했을 때와 보호본으로 학습했을 때 AI가 이 작품을 얼마나 다르게
        흉내내는지 눈으로 직접 비교할 수 있어요. 실제 GPU 작업이라 결과가 나오기까지 10분 이상 걸려요.
      </p>
      <p className="mb-4 rounded border border-neutral-800 bg-neutral-900 px-3 py-3 text-xs text-neutral-500">
        아래 이미지들은 원본을 그대로 보여주는 게 아니라, 학습이 끝난 LoRA가 그때그때 노이즈에서 새로 그려낸
        샘플이에요. 이 작품의 제목("{artwork.title}")을 프롬프트 삼아 학습·생성 양쪽에 똑같이 사용하기 때문에,
        제목이 그림 내용을 잘 설명할수록 결과가 원본과 더 닮아 보여요 — 제목이 내용과 무관하면 두 그룹(원본 학습 vs
        보호본 학습) 모두 원본과 달라 보일 수 있어요. 이 경우에도 두 그룹의 결과가 서로 얼마나 다른지(delta)는 보호
        효과를 그대로 반영합니다.
      </p>

      {storedQuery.isLoading && !jobId && <p className="text-sm text-neutral-400">저장된 결과 확인 중...</p>}

      {!jobId && (
        <button
          onClick={() => start.mutate()}
          disabled={start.isPending}
          className="rounded bg-neutral-100 px-3 py-2 text-sm font-medium text-neutral-900 disabled:opacity-50"
        >
          {start.isPending ? "요청 중..." : job ? "다시 실제로 재학습해서 테스트" : "지금 실제로 재학습해서 테스트"}
        </button>
      )}

      {start.isError && (
        <p className="mt-3 text-sm text-red-400">
          시작하지 못했습니다 ({start.error instanceof Error ? start.error.message : "알 수 없는 오류"}).
        </p>
      )}

      {jobId && (!job || job.status === "queued" || job.status === "processing") && (
        <p className="text-sm text-neutral-400">
          {job?.status === "processing" ? "LoRA 재학습 진행 중..." : "대기 중..."} (자동으로 갱신돼요, 몇 분 정도
          걸려요)
        </p>
      )}

      {job?.status === "failed" && (
        <p className="mt-3 rounded border border-red-900 bg-red-950/40 px-3 py-3 text-sm text-red-300">
          {job.error ?? "테스트에 실패했습니다."}
        </p>
      )}

      {job?.status === "completed" && (
        <div className="mt-3 flex flex-col gap-4">
          {!isLiveJob && "checkedAt" in job && (
            <p className="text-xs text-neutral-500">
              📌 저장된 결과 ({new Date((job as { checkedAt: number }).checkedAt).toLocaleString("ko-KR")}에 실행됨) --
              탭을 닫았다 다시 열어도 이 결과는 남아있어요.
            </p>
          )}
          {job.sd15 && <ArchScoreCard title="SD1.5" result={job.sd15} />}
          {job.sdxl && <ArchScoreCard title="SDXL" result={job.sdxl} />}
          <button
            onClick={() => setJobId(null)}
            className="self-start text-xs text-neutral-500 underline hover:text-neutral-300"
          >
            새로 테스트
          </button>
        </div>
      )}
    </div>
  );
}

function ArchScoreCard({ title, result }: { title: string; result: protection.ScoreProtectionArchResult }) {
  const color =
    result.verdict === "PROTECTED"
      ? "bg-green-900/40 text-green-300"
      : result.verdict === "WEAK"
        ? "bg-amber-900/40 text-amber-300"
        : "bg-neutral-800 text-neutral-400";

  return (
    <div className="rounded border border-neutral-800 px-3 py-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium">{title}</span>
        <span className={`rounded px-2 py-0.5 text-xs font-medium ${color}`}>{SCORE_VERDICT_LABEL[result.verdict]}</span>
      </div>
      <dl className="mb-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
        <dt className="text-neutral-500">원본으로 학습한 LoRA의 유사도</dt>
        <dd className="font-mono text-neutral-300">{result.baselineSimilarity.toFixed(4)}</dd>
        <dt className="text-neutral-500">보호본으로 학습한 LoRA의 유사도</dt>
        <dd className="font-mono text-neutral-300">{result.protectedSimilarity.toFixed(4)}</dd>
        <dt className="text-neutral-500">차이 (delta, 클수록 보호 효과가 큼)</dt>
        <dd className="font-mono text-neutral-300">
          {result.delta >= 0 ? "+" : ""}
          {result.delta.toFixed(4)}
        </dd>
      </dl>
      <div className="grid grid-cols-2 gap-3">
        <SampleGrid label="원본으로 학습 → 생성된 이미지" samples={result.baselineSamples} />
        <SampleGrid label="보호본으로 학습 → 생성된 이미지" samples={result.protectedSamples} />
      </div>
    </div>
  );
}

function SampleGrid({ label, samples }: { label: string; samples: string[] }) {
  return (
    <div>
      <p className="mb-1 text-xs text-neutral-500">{label}</p>
      <div className="grid grid-cols-2 gap-1">
        {samples.map((b64, i) => (
          <img
            key={i}
            src={`data:image/png;base64,${b64}`}
            alt={`${label} 샘플 ${i + 1}`}
            className="aspect-square w-full rounded border border-neutral-800 object-cover"
          />
        ))}
      </div>
    </div>
  );
}

const CASE_STATUS_LABEL: Record<DetectionCase["status"], string> = {
  OPEN: "진행 중",
  EVIDENCE_READY: "증거 확보됨",
  NO_MATCH_FOUND: "매치 없음",
  FAILED: "실패",
  NOTIFIED: "권리자 알림 완료",
  RESOLVED: "해결됨",
  ESCALATED: "에스컬레이션됨",
  // detection-svc can land a case here when the suspect URL itself required
  // a login to fetch (server.py) -- most common trap: testing with an
  // artwork's own DETAIL PAGE url instead of its signed image url. Not a
  // pipeline failure, just an unreachable source.
  AUTH_REQUIRED: "접근 시 로그인 필요 (원본 URL 확인)",
  ACCESS_DENIED: "접근 거부됨 (원본 URL 확인)",
};

function DetectionTest({ artwork }: { artwork: Artwork }) {
  const queryClient = useQueryClient();
  const [caseId, setCaseId] = useState<string | null>(null);
  const [suspectUrl, setSuspectUrl] = useState("");
  const [suspectModelUrl, setSuspectModelUrl] = useState("");
  const [fillError, setFillError] = useState<string | null>(null);

  const scan = useMutation({
    mutationFn: () => detection.scanArtwork(artwork.id),
    onSuccess: (res) => setCaseId(res.caseId),
  });
  const report = useMutation({
    mutationFn: () => detection.reportArtwork(artwork.id, suspectUrl),
    onSuccess: (res) => setCaseId(res.caseId),
  });
  const reportModelLeak = useMutation({
    mutationFn: () => detection.reportModelLeak(artwork.id, suspectModelUrl),
    onSuccess: (res) => setCaseId(res.caseId),
  });

  const caseQuery = useQuery({
    queryKey: ["detectionCase", caseId],
    queryFn: () => detection.getDetectionCase(caseId!),
    enabled: Boolean(caseId),
    refetchInterval: (query) =>
      query.state.data && query.state.data.status !== "OPEN" ? false : 2000,
    // A scan/report result matters even if the user tabbed away while it
    // ran (unlike, say, a feed refresh) -- react-query's own default
    // pauses refetchInterval in a backgrounded tab, which would otherwise
    // leave this stuck showing "진행 중" until the user comes back and
    // triggers a refetch some other way.
    refetchIntervalInBackground: true,
  });

  const evidenceQuery = useQuery({
    queryKey: ["detectionEvidence", caseId],
    queryFn: () => detection.getDetectionEvidence(caseId!),
    enabled: Boolean(caseId) && caseQuery.data?.status === "EVIDENCE_READY",
  });

  async function fillWithOwnUrl() {
    setFillError(null);
    try {
      const { url } = await queryClient.fetchQuery({
        queryKey: ["renderUrlForTest", artwork.id],
        queryFn: () => delivery.getRenderUrl(artwork.id, "logged_in"),
        staleTime: 0,
      });
      setSuspectUrl(url);
    } catch {
      setFillError("작품 URL을 가져오지 못했습니다.");
    }
  }

  return (
    <div className="rounded border border-neutral-800 bg-neutral-950/40 px-4 py-4">
      <p className="mb-4 text-sm text-neutral-400">
        누군가 이 작품을 무단으로 재배포했다고 가정하고, 실제 탐지·증빙 파이프라인(웹 검색, 워터마크 복원, 지문
        대조, C2PA 서명 검증, 온체인 등록 대조, 증거 서명)을 그대로 실행해볼 수 있어요.
      </p>

      <div className="mb-4 flex flex-col gap-3 border-b border-neutral-800 pb-4">
        <div>
          <button
            onClick={() => scan.mutate()}
            disabled={scan.isPending || Boolean(caseId)}
            className="rounded bg-neutral-100 px-3 py-2 text-sm font-medium text-neutral-900 disabled:opacity-50"
          >
            {scan.isPending ? "요청 중..." : "웹에서 자동 검색 (Google Vision)"}
          </button>
          <p className="mt-1 text-xs text-neutral-500">GOOGLE_VISION_API_KEY가 설정되어 있어야 실제 검색이 됩니다.</p>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (suspectUrl.trim() && !caseId) report.mutate();
          }}
          className="flex flex-col gap-2"
        >
          <label className="text-sm">
            의심되는 URL로 직접 신고
            <div className="mt-1 flex gap-2">
              <input
                value={suspectUrl}
                onChange={(e) => setSuspectUrl(e.target.value)}
                placeholder="https://..."
                className="flex-1 rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm"
              />
              <button
                type="button"
                onClick={fillWithOwnUrl}
                className="whitespace-nowrap rounded border border-neutral-700 px-3 py-2 text-sm"
              >
                이 작품 URL로 테스트
              </button>
            </div>
          </label>
          {fillError && <p className="text-xs text-red-400">{fillError}</p>}
          <p className="text-xs text-neutral-500">
            "이 작품 URL로 테스트"를 누르면 이 작품 자체의 보호된 이미지 URL을 채워서, 실제로 그 이미지가
            재배포됐다고 가정하고 전체 파이프라인이 동작하는 걸 확인할 수 있어요.
          </p>
          <button
            type="submit"
            disabled={report.isPending || !suspectUrl.trim() || Boolean(caseId)}
            className="self-start rounded bg-neutral-800 px-3 py-2 text-sm font-medium disabled:opacity-50"
          >
            {report.isPending ? "제출 중..." : "이 URL로 신고 접수"}
          </button>
        </form>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (suspectModelUrl.trim() && !caseId) reportModelLeak.mutate();
          }}
          className="flex flex-col gap-2 border-t border-neutral-800 pt-3"
        >
          <label className="text-sm">
            의심되는 LoRA/모델 파일 URL로 학습 유출 신고
            <input
              value={suspectModelUrl}
              onChange={(e) => setSuspectModelUrl(e.target.value)}
              placeholder="https://civitai.com/models/... 또는 .safetensors 직접 링크"
              className="mt-1 w-full rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm"
            />
          </label>
          <p className="text-xs text-neutral-500">
            이미지가 아니라 "누군가 이 작품으로 LoRA를 학습시켜서 배포했는지"를 확인합니다. 이 작품의 공개된 보호본
            이미지와, 그 모델로 생성한 샘플의 CLIP 유사도를 baseline과 비교해요.
          </p>
          <button
            type="submit"
            disabled={reportModelLeak.isPending || !suspectModelUrl.trim() || Boolean(caseId)}
            className="self-start rounded bg-neutral-800 px-3 py-2 text-sm font-medium disabled:opacity-50"
          >
            {reportModelLeak.isPending ? "제출 중..." : "이 모델로 학습유출 신고 접수"}
          </button>
        </form>
      </div>

      {(scan.isError || report.isError || reportModelLeak.isError) && (
        <p className="mb-4 text-sm text-red-400">요청에 실패했습니다. 다시 시도해주세요.</p>
      )}

      {caseId && (
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-neutral-400">케이스</span>
            <span className="font-mono text-xs">{caseId}</span>
            {caseQuery.data && <CaseStatusBadge status={caseQuery.data.status} />}
            {caseQuery.data?.status === "OPEN" && (
              <span className="text-xs text-neutral-500">진행 중... 자동 갱신됩니다</span>
            )}
          </div>

          {caseQuery.data?.status === "NO_MATCH_FOUND" && (
            <p className="rounded border border-neutral-800 bg-neutral-900 px-3 py-3 text-sm text-neutral-400">
              매치되는 증거를 찾지 못했습니다. "이 작품 URL로 테스트"를 사용했다면, 워터마크/지문이 100% 일치해야 하는
              작품이 아직 완전히 처리되지 않았을 수 있어요.
            </p>
          )}

          {caseQuery.data?.status === "FAILED" && (
            <p className="rounded border border-red-900 bg-red-950/40 px-3 py-3 text-sm text-red-300">
              {caseQuery.data.error_message ?? "케이스 처리 중 오류가 발생했습니다."}
            </p>
          )}

          {evidenceQuery.data?.bundles.map((bundle, i) => (
            <EvidenceCard key={i} bundle={bundle} />
          ))}

          <button
            onClick={() => {
              setCaseId(null);
              setSuspectUrl("");
              setSuspectModelUrl("");
            }}
            className="self-start text-xs text-neutral-500 underline hover:text-neutral-300"
          >
            새 테스트 시작
          </button>
        </div>
      )}
    </div>
  );
}

function CaseStatusBadge({ status }: { status: DetectionCase["status"] }) {
  const color =
    status === "EVIDENCE_READY"
      ? "bg-green-900 text-green-300"
      : status === "FAILED"
        ? "bg-red-900 text-red-300"
        : status === "NO_MATCH_FOUND"
          ? "bg-neutral-800 text-neutral-400"
          : "bg-neutral-800 text-neutral-300";
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${color}`}>{CASE_STATUS_LABEL[status]}</span>;
}

function EvidenceCard({ bundle }: { bundle: EvidenceBundle }) {
  const wm = bundle.watermarkDetection;
  const c2pa = bundle.c2paDetection;
  const modelLeak = bundle.modelLeakDetection;

  if (modelLeak) {
    return (
      <div className="rounded border border-green-900 bg-green-950/20 px-4 py-4">
        <p className="mb-3 text-sm font-medium text-green-300">
          ✓ 학습 유출 의심 — 이 모델로 생성한 이미지가 baseline보다 이 작품에 유의미하게 더 가까워요
        </p>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
          <dt className="text-neutral-500">의심 모델 URL</dt>
          <dd className="break-all text-neutral-300">{bundle.discoveredUrl}</dd>
          <dt className="text-neutral-500">판정</dt>
          <dd className="text-neutral-300">{modelLeak.verdict}</dd>
          <dt className="text-neutral-500">평균 delta (suspect - base)</dt>
          <dd className="font-mono text-neutral-300">
            {modelLeak.meanDelta >= 0 ? "+" : ""}
            {modelLeak.meanDelta.toFixed(4)} (기준값 {modelLeak.threshold})
          </dd>
        </dl>
        {modelLeak.perPrompt.length > 0 && (
          <table className="mt-3 w-full text-xs">
            <thead>
              <tr className="text-neutral-500">
                <th className="pb-1 text-left font-normal">프롬프트</th>
                <th className="pb-1 text-right font-normal">base 유사도</th>
                <th className="pb-1 text-right font-normal">suspect 유사도</th>
                <th className="pb-1 text-right font-normal">delta</th>
              </tr>
            </thead>
            <tbody>
              {modelLeak.perPrompt.map((p, i) => (
                <tr key={i} className="border-t border-green-900/40 text-neutral-300">
                  <td className="py-1 pr-2">{p.prompt}</td>
                  <td className="py-1 text-right font-mono">{p.avgBaseSimilarity.toFixed(4)}</td>
                  <td className="py-1 text-right font-mono">{p.avgSuspectSimilarity.toFixed(4)}</td>
                  <td className="py-1 text-right font-mono">
                    {p.delta >= 0 ? "+" : ""}
                    {p.delta.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    );
  }

  return (
    <div className="rounded border border-green-900 bg-green-950/20 px-4 py-4">
      <p className="mb-3 text-sm font-medium text-green-300">✓ 증거 확보됨 — 실제 파이프라인이 이 URL을 이 작품의 사본으로 판정했어요</p>

      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
        <dt className="text-neutral-500">발견된 URL</dt>
        <dd className="break-all text-neutral-300">{bundle.discoveredUrl}</dd>

        <dt className="text-neutral-500">지문(pHash) 거리</dt>
        <dd className="text-neutral-300">{bundle.phashDistance ?? "측정 안 됨"} (0에 가까울수록 동일)</dd>

        {wm && (
          <>
            <dt className="text-neutral-500">워터마크 매치</dt>
            <dd className={wm.isMatch ? "text-green-300" : "text-neutral-400"}>
              {wm.isMatch ? "일치" : "불일치"} (신뢰도 {(wm.avgConfidence * 100).toFixed(0)}%, 비트오류율{" "}
              {(wm.bitErrorRate * 100).toFixed(1)}%)
            </dd>
          </>
        )}

        {c2pa && (
          <>
            <dt className="text-neutral-500">C2PA 서명</dt>
            <dd className="text-neutral-300">
              {c2pa.hasManifest
                ? c2pa.signedByDontai
                  ? "DONTAI 서명 매니페스트 발견"
                  : "매니페스트는 있으나 DONTAI 서명 아님"
                : "매니페스트 없음"}
            </dd>
          </>
        )}

        {bundle.onchainTransaction?.txHash && (
          <>
            <dt className="text-neutral-500">온체인 등록</dt>
            <dd>
              <a
                href={`https://amoy.polygonscan.com/tx/${bundle.onchainTransaction.txHash}`}
                target="_blank"
                rel="noreferrer"
                className="text-blue-400 underline"
              >
                Polygon Amoy 익스플로러에서 보기 ↗
              </a>
            </dd>
          </>
        )}

        {bundle.signature && (
          <>
            <dt className="text-neutral-500">증거 서명</dt>
            <dd className="text-neutral-300">{bundle.signature.algorithm} (KMS 연동 서명 완료)</dd>
          </>
        )}
      </dl>
    </div>
  );
}
