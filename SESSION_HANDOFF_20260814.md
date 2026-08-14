# 세션 작업 정리 (2026-08-13 ~ 2026-08-14)

다른 세션에서 이어받을 수 있도록, 이번 세션에서 실제로 바뀐 것들을 정리한 문서입니다.
전부 Pi(`philosophyz@Philosophyz.iptime.org`, `/media/philosophyz/SSD/dontai/`)에 배포·재기동까지 완료된 상태입니다.

## 1. 프로덕션 보호 메커니즘 교체 (색감 훼손 수정)

- **문제**: 프로덕션이 `hybrid_protect.py`(SD1.5+SDXL 둘 다 latent 스테이지 먼저 실행)를 쓰고 있었고, 이게 실사용 이미지에서
  오일페인팅 왜곡·색감 붕괴(마젠타/그린 얼룩)를 일으키는 걸로 확인됨. 이 모듈 자체가 "n=30 검증 전 프로덕션 투입 금지"라고
  스스로 문서화해놨는데도 배포돼 있었음.
- **해결**: `apps/protection-svc/ml-engine/src/clean_protect.py` 신규 작성 — latent 스테이지 없이 순수 pixel-space 4단계
  체인(SD1.5 full → SDXL full → SD1.5 top-up → SDXL top-up). `handler.py`의 기본 액션을 `clean_cloak`으로 변경.
- **체크보드/얼룩무늬 근본원인 발견**: SDXL이 `param_size`(저해상도 그리드)에서 최적화한 delta를 원본 해상도로
  bicubic 업샘플링하는 과정 자체가 앨리어싱을 만듦. `param_size`를 1024→1536으로 올려서 해결 (강도 delta_sdxl
  0.0934→0.0559로 다소 낮아지지만 화질 깨끗함).
- **측정치(n=1)**: delta_sd15 +0.1097, delta_sdxl +0.1035.
- **주의**: 프로덕션 SDXL 체크포인트는 vanilla SDXL이 아니라 Illustrious-XL(애니메이션 파인튠) — 실측치가 다를 수 있음.
  n=1 검증이라 n=30 규모 재현은 아직 안 됨.
- Docker 이미지도 재빌드·재푸시 필요했음(Pi는 GPU 없음, 실제 연산은 RunPod Serverless가 pull하는
  `lentropy/dontai-strongprotect-serverless:latest` 이미지에서 실행됨).

## 2. 프로덕션 크롭/줌 버그 (사용자가 직접 발견)

- `orchestrate.py`의 "원본 해상도 복구" 블록이 `used_strong_protection` 여부와 무관하게 항상 실행되고 있었음.
  `clean_protect.py` 출력은 이미 원본 해상도인데, 옛날 `cloak()`용 letterbox-crop 로직이 다시 적용되면서 이미지
  일부만 잘라내 확대한 것처럼 보이는 버그였음.
- `if not used_strong_protection:`으로 조건부 처리하도록 수정 + 비정사각형 fixture로 회귀 테스트 추가
  (`test_orchestrate.py`).

## 3. RunPod 인프라

- **executionTimeoutMs 버그**: `dontai-strongprotect` 엔드포인트가 30분 타임아웃인데 clean_protect.py 4단계
  체인은 그보다 오래 걸려서 stage 2 도중 서버 사이드 타임아웃으로 죽었음. 50분(3000000ms)으로 상향 + 클라이언트
  타임아웃도 3300s로 동반 상향.
- **청구 내역 점검**: 최근 3주 RunPod 지출 $15.13 — 전부 실제 작업일과 일치, IDLE 워커는 며칠 떠 있어도 과금
  안 됨(실행 시간에만 과금) 확인. 새는 돈 없음.
- **Serverless 동시성 게이트 추가** (`apps/protection-svc/remote_gpu.py`): `/protect`, `/score-protection`,
  `/lora-jobs`가 전부 같은 `dontai-strongprotect` 엔드포인트(워커 최대 2개)로 몰릴 수 있었는데, 프로세스 내
  ThreadPoolExecutor만으론 RunPod 쪽 실제 워커 capacity를 인식 못 했음. `threading.Semaphore` 기반 엔드포인트별
  게이트를 추가해서 초과 요청은 로컬에서 대기(jobs_db엔 정직하게 "processing"으로 표시)하도록 함.
  (`RUNPOD_STRONGPROTECT_MAX_CONCURRENT`/`RUNPOD_STYLECLOAK_MAX_CONCURRENT` 환경변수로 조정 가능, 기본 2)

## 4. motection-changes 팀원 패치 병합

팀원이 `motection-changes-20260812-112347/`에 넘겨준 작업을 검토 후 병합:
- **detection-svc**: 기존 프로젝트의 더 발전된 기능(blockchain_client, dmca_notice, evidence_signing 등)은
  유지하면서 팀원의 새 기능(monitor_scheduler, perceptual_hash, phash_index, robust_fingerprint, verdict,
  evidence_integrity)을 추가.
- **delivery-gateway**: `bot_policy.rs`(신규) + `crawlers.rs` 확장(BotGroup/BotIdentity/classify) — 아트워크별
  봇 ALLOW/BLOCK/LOG_ONLY 정책. 처음엔 **프로세스 메모리에만 저장**되는 상태로 병합됐었음(팀원이 "DB 연동만 하면
  완성"이라고 했던 부분) → §5에서 실제로 DB 영속화함.
- **asset-service**: `tools/seed_detection_artworks.py` 추가.

## 5. 봇 정책/로그 DB 영속화 + 허용·차단 UI (이번 세션에 완성)

팀원이 "DB 연동만 하면 완성"이라고 했던 부분이 실제로는 안 되어 있었음 (§4에서 병합된 코드는 재시작하면
정책·로그가 전부 날아가는 상태였음). 그리고 사이트에 허용/차단 UI 자체가 없었음. 둘 다 이번에 구현:

- **asset-service**: `bot_policies`, `bot_access_logs` 테이블 추가 (`drizzle/0011`, `0012`).
  `GET/PUT /artworks/:id/bot-policy`, `GET/POST /artworks/bot-access-logs`, `GET /artworks/bot-policies`(벌크,
  delivery-gateway 시작 시 캐시 하이드레이션용) 라우트 추가.
- **delivery-gateway**: `put_bot_policy`가 asset-service에 write-through 한 뒤에만 로컬 캐시 갱신. 봇 접근
  기록 시 asset-service로 fire-and-forget 비동기 push (`spawn_bot_access_log_push`). 프로세스 시작 시
  `hydrate_bot_policies()`로 기존 정책 전체를 asset-service에서 로드.
- **web**: `ArtworkDetailPage`에 창작자 전용 `BotPolicyPanel` 추가 — 봇 그룹별(검색엔진/AI학습/AI검색/AI어시스턴트/
  SNS미리보기/SEO/미분류) ALLOW/BLOCK/LOG_ONLY 지정 + 최근 접근 기록 조회.

## 6. 코인 "원본 보기" 로직 재설계

기존 설계 문제: 창작자가 먼저 "원본 미리보기 공개" 토글을 켜야만 코인 버튼이 뜨는 2단계 구조였음(원래 의도와 다름).
사용자 확인 결과:
- 사전 토글 없이 **기본적으로 항상 코인 버튼 노출** (창작자가 막고 싶을 때만 끄는 방식으로 반전).
- 코인 내고 보는 건 여전히 **다운스케일+워터마크 근사치**(진짜 원본 아님, 유출 방지 설계 유지).

변경 내역:
- `artworks.originalPreviewBlocked` 컬럼 추가 (기본 false). 창작자 전용 `PUT /:id/original-preview-blocked`로
  on/off.
- 언락 시 파생 이미지가 없으면 그 자리에서 생성(`ensureOriginalPreviewGenerated`) — 사전 생성 불필요.
- api-gateway/web 전부 동기화 (`setOriginalPreviewBlocked`, `enableOriginalPreview`/`disableOriginalPreview`
  제거).
- web UI: 창작자에겐 "코인으로도 원본 공개 금지" 체크박스, 뷰어에겐 (막혀있지 않으면) 항상 코인 버튼 표시.

## 7. 컴플라이언스 폴더 (`complience/`) 검토 및 적용

팀원이 넘긴 5개 문서(ToS 초안, EvidenceGenerator.js, Audit Log DB 스키마, gitleaks CI, 응답 페이로드 예시) 검토 결과:

- **증거 패키지(EvidenceGenerator.js)**: 이미 프로젝트에 더 나은 버전이 실제로 동작 중이었음
  (`detection-svc/src/evidence_bundle.py` + `evidence_signing.py` — 실제 Ed25519 서명, KMS 봉투암호화 키 사용,
  Test Lab에서 다운로드 가능). 팀원 초안은 적용 안 함(중복).
- **WORM 관련 응답 예시(`isLockedWORM: true`)**: 실제 WORM 스토리지는 아직 미구현(팀원 문서 자체도
  "논의만 진행, 구현 보류"라고 명시) — 없는 기능을 있는 것처럼 노출하지 않기 위해 적용 안 함.
- **적용한 것**:
  - `.github/workflows/security-scan.yml` — gitleaks 시크릿 스캔 CI 신규 추가.
  - `compliance_audit_logs` 테이블 신규 추가 (`drizzle/0013`) — Postgres `CREATE RULE` 초안을 SQLite
    트리거(`BEFORE UPDATE/DELETE ... RAISE(ABORT, ...)`)로 변환해 append-only 강제. 업로드 시 AI학습 동의,
    봇 정책 변경, 원본 공개 차단 설정 변경을 기록 (`src/lib/auditLog.ts`).
  - `/terms` 페이지 신규 — Do-Not-Train 조항 + 기술적 보호조치 면책 조항을 실제 시스템 동작에 맞게 다듬어서
    창작자가 확인할 수 있게 노출 (NavBar에 "이용약관" 링크 추가).

## 8. Test Lab i2i → t2i + ControlNet

- i2i→t2i 전환은 이미 되어 있었음(재확인만 함) — `protection_score.py`가 이미 순수 노이즈(`torch.randn`)에서
  시작하고, 업로드 시 태그를 프롬프트로 사용 중이었음.
- **ControlNet 신규 추가** (`ml-engine/src/controlnet_condition.py`): PIL 기반 Canny 엣지맵으로 baseline/protected
  두 미리보기 생성을 원본과 같은 구도로 고정 — 보호 효과 차이가 랜덤 시드 구도 차이 때문이 아니라는 걸 보장.

## 9. 알려진 미해결/열린 이슈

- `rust-core/target/`(1.8GB) — Pi에 cargo가 로그인 셸(`bash -lc`)에선 잡히는 걸 확인했지만, 재빌드 가능 여부는
  실제로 시도 안 해봄. 삭제 안 함.
- clean_protect.py는 n=1 검증만 됨 — n=30 규모 재현은 사용자 승인 대기 중.
- 실제 KMS(외부 서비스)/WORM 스토리지는 여전히 미구현 — 이번 세션에서도 손대지 않음(팀원 스코프 밖으로 명시된 부분).
- Task #24 (Sequential SD1.5→SDXL native chain experiment)는 예전부터 pending 상태로 남아있음.

## 배포 상태

이번 세션에서 변경된 모든 서비스(asset-service, delivery-gateway, api-gateway, web, protection-svc)는
Pi에 코드 배포 + 빌드(delivery-gateway는 `cargo build --release`, api-gateway/web은 `npm run build`) +
마이그레이션(`npm run db:migrate`) + `systemctl restart` 까지 완료된 상태입니다.
