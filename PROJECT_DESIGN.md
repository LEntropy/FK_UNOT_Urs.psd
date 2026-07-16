# AI 학습 방지 · 저작권 보호 창작 플랫폼 — 설계 문서

> 코드명: **DONTAI** (Do Not Train AI)
> 문서 버전: v1.0 · 작성일: 2026-07-01
> 대상: 기획/개발 5인 팀

---

## 0. 이 문서가 하는 일

네가 준 개요(`AI_DRM_정리.md`)는 **기업용(B2B) DRM 솔루션**을 전제로 쓰여 있었다.
하지만 네가 실제로 만들려는 건 **PIXIV 같은 창작자 커뮤니티(B2C)** 에 다음을 얹은 서비스다.

- AI가 그림체를 무단 학습하지 못하게 하는 **보호 변환**
- **블록체인 기반 소유권 증명**
- 침해 시 **법적 대응 자료 제공**
- 준비된 **C 언어 KMS 서버**를 이용한 암호화/컴플라이언스

그래서 이 문서는 참고 자료를 그대로 쓰지 않고 **B2C 커뮤니티 + 블록체인 + 5인 팀 규모**에 맞게 재구성했다.

### 0-1. 참고 문서에서 "덜어낸" 것

| 덜어낸 항목 | 이유 |
|---|---|
| 멀티테넌트(Org/Tenant) 전체 구조 | 우리는 개인 창작자 대상 커뮤니티. `조직` 대신 `유저`가 최상위 |
| SSO(SAML/OIDC), RBAC+ABAC 기업 권한 모델 | 소셜 로그인 + 단순 역할(유저/크리에이터/운영자)로 충분 |
| 파트너 라이선스 배포/기업 계약 워크플로우 | B2C 초기엔 불필요. 라이선스는 "이용 약관 + CC 라이선스 표기" 수준으로 시작 |
| Kafka 급 대형 메시지 버스, OpenSearch 필수화 | 5인 팀·초기 트래픽엔 과설계. Redis Queue/BullMQ로 시작 |
| Billing/과금 콘솔 | 수익화는 후순위. MVP에서 제외 |

### 0-2. 참고 문서에 "보완/추가" 한 것

| 추가 항목 | 이유 |
|---|---|
| **블록체인 소유권 레지스트리** (참고 문서엔 사실상 없음) | 네 핵심 차별점. 온체인 앵커링 설계를 새로 추가 |
| **커뮤니티 기능**(피드, 팔로우, 태그, 좋아요, 컬렉션) | PIXIV 벤치마킹의 핵심. 참고 문서엔 전혀 없었음 |
| **C 언어 KMS 연동 설계** | 이미 준비된 자산. 서명/암호화 경계를 명확히 정의 |
| **침해 대응 매뉴얼(런북)** | "법적 레거시 준비"라는 목표를 실행 절차로 구체화 |
| **5인 팀 역할 분담 + 현실적 로드맵** | 실제 실행 가능성 확보 |

### 0-2-1. 프로젝트 성격 (로드맵/의사결정의 기준)

이 프로젝트는 **포트폴리오용 학교 프로젝트**이지만, 목표는 **"실서비스 직전(near-production) 수준"** 까지 완성하는 것이다. 이 성격이 아래 결정들을 지배한다.

- **완성도 > 신기능 개수**: 반쯤 되는 기능 10개보다, 끝까지 도는 핵심 플로우 1개(업로드→보호→온체인→갤러리→탐지)가 포트폴리오에서 훨씬 강하다.
- **프로덕션 관행은 지키되, 비용은 아낀다**: CI, 테스트, 시크릿 관리, 감사 로그, mTLS 같은 "실서비스 관행"은 갖춘다. 단 블록체인은 **테스트넷(Amoy) 유지**(메인넷은 스트레치 목표), 인프라는 무료/저가 티어(Fly.io, Supabase 등)로.
- **데모 임팩트 우선**: 심사/면접에서 3분 안에 "온체인 소유권 증명 + 보호 전/후 화풍 모방 실측 비교"를 보여줄 수 있는 시나리오를 Phase마다 하나씩 확보.
- **문서·다이어그램이 곧 산출물**: 아키텍처/보안 컴플라이언스 문서 자체가 포트폴리오 가치가 크므로 계속 최신화한다.

### 0-3. 한 줄 제품 정의

> **"창작자가 그림을 올리면 → 보호 변환(AI 학습 방해) → 블록체인에 소유권 등록 → 안전하게 공개 → 무단 학습/도용 탐지 → 침해 시 증거 패키지 제공"** 까지 이어지는, AI 시대 창작자 보호 커뮤니티.

핵심 메시지는 **"완전 차단"이 아니라 "억제 · 추적 · 증빙 · 학습 경제성 파괴"** 다. (100% 차단은 기술적으로 불가능하다는 점을 팀 전체가 공유해야 한다 — §12 참고.)

---

## 1. 서비스 구성도 (사용자 관점 / 논리 구조)

사용자와 데이터가 어떻게 흐르는지를 보여주는 큰 그림이다.

```
                          ┌───────────────────────────────────┐
                          │            사용자 (브라우저)          │
                          │  크리에이터 · 감상자 · 운영자          │
                          └───────────────┬───────────────────┘
                                          │  HTTPS
                          ┌───────────────▼───────────────────┐
                          │      Web Frontend (React + TS)      │
                          │  갤러리 · 업로드 · 크리에이터 대시보드   │
                          └───────────────┬───────────────────┘
                                          │  REST / WebSocket
                    ┌─────────────────────▼─────────────────────┐
                    │        API Gateway / Auth (Node.js)         │
                    │  인증 · 라우팅 · 레이트리밋 · 정책 판정         │
                    └───┬──────────┬──────────┬──────────┬───────┘
                        │          │          │          │
          ┌─────────────▼──┐ ┌─────▼─────┐ ┌──▼───────┐ ┌▼──────────────┐
          │  Asset/Community│ │ Protection │ │ Delivery │ │  Blockchain   │
          │   Service       │ │  Pipeline  │ │ Gateway  │ │   Service     │
          │ (Node)          │ │(Rust+Py ML)│ │ (Rust)   │ │ (Node/Rust)   │
          │ 작품·피드·태그    │ │ 보호변환    │ │ 조건부배포 │ │ 소유권 온체인   │
          │ 팔로우·좋아요     │ │ 워터마크    │ │ 안티스크랩 │ │ 등록·검증      │
          └───────┬────────┘ └─────┬──────┘ └────┬─────┘ └──────┬────────┘
                  │                │             │              │
                  │          ┌─────▼─────┐       │        ┌─────▼──────┐
                  │          │  KMS (C)  │◄──────┼────────┤ 블록체인 노드 │
                  │          │ 암호화/서명 │       │        │ (Polygon L2)│
                  │          └───────────┘       │        └────────────┘
                  │                              │
     ┌────────────▼──────────────────────────────▼─────────────────┐
     │                        데이터 계층                            │
     │  PostgreSQL(메타/권리)  ·  Redis(큐/캐시)  ·  Object Store     │
     │  ┌─────────────┬──────────────┬─────────────────────────┐    │
     │  │ 원본 저장소   │  보호본 저장소  │  증거 저장소(WORM/버전)    │    │
     │  │ (암호화·비공개)│ (공개/썸네일)   │ (탐지 스냅샷·감사로그)     │    │
     │  └─────────────┴──────────────┴─────────────────────────┘    │
     └───────────────────────────────────────────────────────────────┘
                  ▲
                  │ 주기적 스캔
     ┌────────────┴──────────────────────────────────────┐
     │       Monitoring & Detection (Python)              │
     │  역이미지 검색 · pHash 유사도 · 워터마크 검출         │
     │  → 침해 의심 발견 시 Evidence 저장소에 증거 패키지 생성  │
     └───────────────────────────────────────────────────┘
```

### 1-1. 핵심 사용자 흐름 (업로드 → 공개까지)

```
크리에이터 업로드
   │
   ├─▶ 1. 파일 검증 (형식/크기/악성)
   ├─▶ 2. 원본 SHA-256 + perceptual hash(pHash) 생성
   ├─▶ 3. KMS로 원본 암호화 → 원본 저장소(비공개)
   ├─▶ 4. 보호 프로파일 선택 (LEVEL 1~4)
   ├─▶ 5. 보호 파이프라인 실행
   │        ├ Style Confusion (화풍 교란)
   │        ├ Concept Misalignment (개념 정렬 교란, 상위 레벨)
   │        ├ Invisible Watermark 삽입
   │        └ C2PA 출처 메타데이터 부여
   ├─▶ 6. 보호본 저장소에 공개용/썸네일 버전 저장
   ├─▶ 7. 블록체인에 소유권 앵커 등록
   │        (pHash + 메타해시 + 지갑주소 + 타임스탬프 → 온체인)
   │        └ 트랜잭션 해시를 작품 레코드에 저장
   └─▶ 8. 갤러리에 공개 (Delivery Gateway 경유 signed URL로만 노출)
```

---

## 2. 개발 구성도 (기술 관점 / 컴포넌트·저장소 구조)

실제로 **어떤 언어로 어떤 서비스를 만들고, 리포지토리를 어떻게 나눌지**를 보여준다.

```
┌──────────────────────────────────────────────────────────────────┐
│                        Monorepo (GitHub)                           │
│  /apps                                                             │
│   ├─ web/                React + TypeScript + Vite        [FE]      │
│   │                      TanStack Query, Zustand, Tailwind         │
│   ├─ api-gateway/        Node.js (NestJS or Fastify)     [BE Core] │
│   │                      인증·라우팅·레이트리밋                       │
│   ├─ asset-service/      Node.js (NestJS)               [BE Core]  │
│   │                      작품/커뮤니티/피드 CRUD                     │
│   ├─ protection-svc/     Rust (axum) + Python(FastAPI)   [Systems/AI]│
│   │   ├─ rust-core/      빠른 이미지 변환·워터마크·해시              │
│   │   └─ ml-engine/      PyTorch: style confusion / misalignment  │
│   ├─ delivery-gateway/   Rust (axum + tower)             [Systems] │
│   │                      signed URL·안티스크랩·조건부 렌더           │
│   ├─ blockchain-svc/     Node.js (ethers.js) or Rust     [Chain]   │
│   │                      스마트컨트랙트 연동·온체인 등록/검증          │
│   ├─ detection-svc/      Python (FastAPI + Celery)       [AI]      │
│   │                      역이미지 검색·유사도·증거 생성              │
│   └─ contracts/          Solidity (Hardhat/Foundry)      [Chain]   │
│                          OwnershipRegistry.sol                     │
│                                                                    │
│  /packages (공유)                                                  │
│   ├─ shared-types/       TS 타입 (API 계약, DTO)                    │
│   ├─ proto/              gRPC/protobuf 정의 (내부 서비스 간)         │
│   └─ ui/                 공용 React 컴포넌트                        │
│                                                                    │
│  /infra                                                            │
│   ├─ docker-compose.yml  로컬 개발 전체 스택                        │
│   ├─ k8s/ or fly/        배포 매니페스트                            │
│   └─ kms-adapter/        C KMS 서버 연동 래퍼(gRPC/소켓)            │
└──────────────────────────────────────────────────────────────────┘

           내부 통신                         외부 의존
   ┌──────────────────────┐        ┌──────────────────────────┐
   │ 동기: gRPC / HTTP     │        │ Polygon RPC (Alchemy 등)  │
   │ 비동기: Redis(BullMQ) │        │ C2PA 라이브러리            │
   │ 캐시: Redis           │        │ 역이미지검색 API/자체 인덱스 │
   └──────────────────────┘        │ C KMS 서버 (사내)          │
                                    └──────────────────────────┘
```

### 2-1. 왜 이렇게 언어를 나눴나 (네 스택 근거)

| 서비스 | 언어 | 이유 |
|---|---|---|
| web | React + TypeScript | 요구사항 그대로. Vite로 빠른 DX |
| api-gateway / asset-service | Node.js | 요구사항. I/O 바운드 CRUD·인증에 적합, 생태계 풍부 |
| protection-svc (core) | **Rust** | 이미지 픽셀 변환·워터마크·해시는 CPU 바운드 → Rust로 처리량 확보 |
| protection-svc (ml) | **Python** | Glaze/Nightshade 계열은 PyTorch 기반. ML은 파이썬이 사실상 표준 |
| delivery-gateway | **Rust** | 고빈도 요청·저지연·안티스크랩 판정 → Rust가 강점 |
| blockchain-svc | Node.js(ethers.js) | 스마트컨트랙트 연동 툴체인이 JS/TS에 가장 성숙 |
| KMS | **C (준비됨)** | 이미 완성. 우리는 gRPC/소켓 어댑터만 붙임 |

> 팁: Rust와 Python이 한 서비스(protection-svc)에 공존한다. Rust가 전처리/후처리(빠름)를 맡고, 무거운 ML 변환만 Python 서브프로세스/gRPC로 호출하는 구조를 권장. 초기엔 **전부 Python으로 시작 → 병목 구간만 Rust로 이관**해도 된다(조기 최적화 회피).

---

## 3. 모듈별 상세 설계

### 3-1. Auth / 사용자
- 소셜 로그인(Google/Kakao) + 이메일. JWT(access) + refresh 토큰.
- **지갑 연결**: 블록체인 소유권 등록을 위해 크리에이터는 지갑(MetaMask 등)을 연결하거나, 초기엔 **플랫폼 커스터디 지갑**(서버가 대리 서명, KMS로 키 보관)을 제공.
- 역할: `USER`(감상), `CREATOR`(업로드), `MODERATOR`, `ADMIN`.

### 3-2. Asset / Community Service
작품 메타데이터 + 커뮤니티 기능의 중심. 주요 엔티티는 §4 참고.
- 작품 CRUD, 태그, 시리즈/컬렉션
- 피드(팔로잉/인기/최신), 좋아요·북마크·팔로우
- 신고(Report) 접수 → Moderation 큐

### 3-3. Protection Pipeline (핵심 방어 엔진)
4개 층으로 구성. **원본은 절대 변형하지 않고, 공개 배포본에만 적용**한다.

```
[1] Perceptual-Preserving Transform  사람 눈엔 거의 동일, 특징추출기엔 교란
       고주파 위상 교란 · 국소 대비 재분배 · 엣지 방향 섭동 · 다중스케일 잔차
[2] Style Confusion Layer            화풍 벡터를 다른 스타일 공간으로 밀어냄 (Glaze 계열)
       목표: 스타일 LoRA 품질 저하, "이 작가 느낌" 재현도 하락
[3] Concept Misalignment Layer       텍스트-이미지 정렬 교란 (Nightshade 계열, 상위 레벨만)
       목표: fine-tuning 시 prompt alignment 불안정, 샘플 효율 저하
[4] Traceability Layer               invisible watermark + robust fingerprint + C2PA manifest
       목표: 사후 추적·법적 증빙
```

최적화 철학(수식적 직관):
```
maximize   Feature_Drift + Style_Misalignment
subject to Perceptual_Distance  < ε        (사람 눈 품질 유지)
           Print/SNS_Quality_Loss < θ      (인쇄/압축 후 품질 손상 제한)
```

> 현실 목표는 "모델 붕괴"가 아니라 **"학습 ROI 저하"**. 강하게 걸수록 색 밴딩·노이즈 부작용이 커지므로 레벨을 옵트인으로 둔다.

### 3-4. 보호 강도 프리셋

| 프리셋 | 워터마크 | Style Confusion | Concept Misalign | 원본 다운로드 | 용도 |
|---|---|---|---|---|---|
| `L1_PREVIEW` | 약 | 약 | 없음 | 불가 | SNS/썸네일 공개 |
| `L2_PORTFOLIO` | 중 | 중 | 약 | 불가 | 일반 공개 포트폴리오 |
| `L3_ANTI_TRAIN` | 강 | 강 | 중~강 | 불가 | 무단 학습 억제 최우선 |
| `L4_LICENSED` | 강 | — | — | 허용(추적) | 라이선스 구매자에게 원본 전달 |

### 3-5. Delivery Gateway (Rust)
- **영구 URL 금지 → 정책 기반 signed URL**: `/asset/{id}/render?variant=public&token=...`
- 짧은 TTL, referer 검증, UA 분류, rate limit, hotlink 차단
- 접속 상황별 다른 변형본:
  - 비로그인 → 1280px 보호본
  - 로그인 유저 → 2048px 보호본
  - 의심 크롤러 → 차단 또는 decoy(허니) 응답
- robots.txt / `X-Robots-Tag: noindex, noimageindex` 자동 생성 + AI 크롤러 denylist (GPTBot, OAI-SearchBot, Google-Extended, ClaudeBot 등)
  - ⚠️ robots는 **협조형 통제**일 뿐 강제력 없음. 실제 방어는 게이트웨이 접근 통제가 담당.

### 3-6. Blockchain Service (소유권 증명)
설계 결정은 §5에서 상세히.

### 3-7. Monitoring & Detection (Python)
- pHash 유사도 검색 + 역이미지 검색(외부 API 또는 자체 인덱스)
- 워터마크/지문 검출
- 발견 시 **증거 패키지** 생성 → Evidence 저장소(WORM)
- 증거 = { 원본해시, 보호본해시, 등록시각, 권리자, 워터마크검출, 발견URL, 발견시각, 스크린샷, HTTP헤더, 온체인 트랜잭션, 내부 서명 } → PDF + JSON 번들

### 3-8. asset-service ↔ protection-svc 통합 지점

protection-svc도 blockchain-svc와 같은 원칙: 이미지·프로파일만 받아서 보호본+해시를 돌려주는 순수 서비스다. asset-service가 오케스트레이터. 실제 API 계약, preset→파라미터 매핑, 그리고 실측으로 확인된 한계는 [apps/protection-svc/INTEGRATION.md](../apps/protection-svc/INTEGRATION.md)에 정리했고, 여기서는 핵심만 요약한다.

**호출 시점**: §1-1 업로드 파이프라인의 "보호 파이프라인 실행" 단계. `POST /protect`로 시작하고, 처리시간이 GPU 기준 수십 초~분 단위(§3-3의 4개 레이어 중 style confusion만으로도 이 정도)라 **작업(job) 방식**이지 동기 호출이 아니다 — blockchain-svc 연동과 동일한 원칙.

**asset-service가 책임지는 것**:
- `POST /protect` 호출 후 잡 상태를 폴링(또는 웹훅)하고, `completed` 응답의 `perceptualHash`/`metadataHash`를 그대로 `blockchain-svc`의 `POST /assets/register`에 전달 (필드명을 일치시켜서 재계산 없이 바로 pass-through)
- `protectedImageUri`를 `asset_versions` 테이블에 저장, `artworks.protection_profile`과 실제 적용된 `appliedPreset`이 일치하는지 검증

**실측으로 확인된 중요한 제약** (ml-engine·rust-core PoC 실험 결과, INTEGRATION.md 상세):
- **style-confusion 보호는 원본 해상도의 약 0.3배 밑에서 붕괴한다** — EOT로 학습을 아무리 시켜도 안 됨(정보이론적 한계, ml-engine/README.md 실험 참고)
- **워터마크(rust-core)도 0.25배 리사이즈에서 깨진다** — 단, 이유는 완전히 다름(정보이론적 한계가 아니라 8x8 블록 그리드가 리사이즈로 어긋나는 기하학적 문제, rust-core/README.md 실측). **두 메커니즘의 실패 지점이 겹친다**는 뜻 — "작은 썸네일은 워터마크로 대체 보호된다"는 가정은 틀렸음이 실측으로 확인됨. Delivery Gateway(§3-5)가 2000px 원본에서 150px 그리드 썸네일을 만들면(~0.075배) 그 썸네일엔 두 보호 모두 사실상 없다고 봐야 함
- **perceptualHash 함수 구현 완료** — `ml-engine/src/perceptual_hash.py`, 표준 DCT pHash(`imagehash` 라이브러리) + `hash_size=16`으로 blockchain-svc가 요구하는 32바이트에 정확히 맞춤
- **GPU 의존성**: 현재 `ml-engine/remote/`의 SSH 오프로드는 개발용일 뿐, 프로덕션은 별도 GPU 워커 풀/큐 소비자가 필요
- **오케스트레이션 완료 + 실제 발견**: `apps/protection-svc/orchestrate.py`가 ml-engine cloak → rust-core watermark → rust-core variants → perceptualHash/metadataHash → blockchain-svc 온체인 등록까지 실제로 한 번에 연결해서 검증함(반고흐 명화로 실행, Amoy 테스트넷에 실제 등록 성공). 이 과정에서 개별 컴포넌트 테스트로는 안 보이던 문제 발견: **ml-engine의 처리 해상도가 256x256으로 고정되어 있어서 공개본이 항상 256px가 되고, `public_preview_1280`/`2048` 티어는 원천적으로 도달 불가능** — `size`는 실제로 조절 가능한 파라미터지만(하드 제약 아님), 지금까지 실측한 모든 수치(프리셋 epsilon, EOT 견고성, 0.5x/0.25x 붕괴 지점)가 256px 기준이라 더 큰 해상도는 **검증 안 된 영역**. `orchestrate.py`의 `sizeValidated` 플래그로 이를 명시함

---

## 4. 데이터 모델 (PostgreSQL, B2C + 블록체인 반영)

```sql
users (
  id, email, handle, display_name, avatar_uri,
  role,                       -- USER | CREATOR | MODERATOR | ADMIN
  wallet_address,             -- 온체인 소유권 등록용(custodial or self)
  created_at, status
)

artworks (
  id, creator_id,
  title, description, tags[],
  asset_type,                 -- illustration | comic | ...
  original_sha256,            -- 원본 해시(비공개 원본)
  perceptual_hash,            -- pHash(유사도 탐지용)
  protection_profile,         -- L1_PREVIEW ... L4_LICENSED
  visibility,                 -- public | followers | private
  onchain_tx_hash,            -- 소유권 등록 트랜잭션
  onchain_token_id,           -- 레지스트리 토큰/레코드 id
  c2pa_manifest_uri,
  status,                     -- PROCESSING | PUBLISHED | TAKEDOWN
  created_at, published_at
)

artwork_versions (
  id, artwork_id, variant,    -- original | public_2048 | thumb_512
  storage_uri, is_protected, is_original,
  width, height, format, color_profile, created_at
)

-- 커뮤니티
follows      (follower_id, creator_id, created_at)
likes        (user_id, artwork_id, created_at)
bookmarks    (user_id, artwork_id, collection_id, created_at)
collections  (id, user_id, name, is_public)
comments     (id, artwork_id, user_id, body, created_at)

-- 소유권 / 온체인
ownership_records (
  id, artwork_id,
  owner_wallet, content_hash,     -- 온체인에 앵커된 해시
  chain, contract_address, tx_hash, block_number,
  registered_at, verified_at
)

-- 라이선스 (B2C 최소)
licenses (
  id, artwork_id, license_type,   -- ALL_RIGHTS | CC_BY | CC_BY_NC ...
  allow_ai_training,              -- 기본 false (Do-Not-Train 플래그)
  allowed_use, created_at
)

-- 침해 탐지 / 증거
evidence_records (
  id, artwork_id, evidence_type,  -- reverse_image | dataset_hit | hotlink
  source_url, confidence, artifact_uri, detected_at, status
)

-- 신고 / 운영
reports   (id, reporter_id, artwork_id, reason, status, created_at)
audit_logs(id, actor_id, action, target_type, target_id, metadata_json, created_at)  -- immutable
```

---

## 5. 블록체인 설계 (핵심 차별점 — 참고 문서에 없던 부분)

### 5-1. 무엇을 온체인에 올리나 (원칙)
이미지 자체를 온체인에 올리지 않는다(비용·프라이버시). **"존재/소유 증명 앵커"만** 올린다.

온체인 저장:
- `contentHash` = keccak256(perceptualHash ‖ metadataHash)
- `owner` = 크리에이터 지갑 주소
- `timestamp` (블록 시각)

오프체인 저장(PostgreSQL / Object Store / IPFS 선택):
- 실제 이미지(암호화 원본 + 보호본), 메타데이터, C2PA manifest

### 5-2. 체인 선택 (권장)

| 후보 | 장점 | 단점 | 판단 |
|---|---|---|---|
| **Polygon PoS / L2** | 수수료 저렴, EVM 생태계, ethers.js 성숙 | 체인 신뢰 이슈 존재 | ✅ **1순위 권장** |
| Ethereum L1 | 최고 신뢰 | 가스비 비쌈 | ❌ 초기 부적합 |
| Private/PoA 체인 | 무료·통제 가능 | "탈중앙 증명" 명분 약함 | 데모/내부용 대안 |
| 해시 앵커링만(OpenTimestamps) | 초저비용 | 소유권 이전·조회 기능 없음 | 보조 수단 |

> 결론: **Polygon(Amoy 테스트넷 → 메인넷)** 위에 커스텀 레지스트리 컨트랙트. 학습/데모 단계는 테스트넷으로 무료 진행.

### 5-3. 스마트컨트랙트 스케치 (Solidity)

```solidity
// contracts/OwnershipRegistry.sol
pragma solidity ^0.8.24;

contract OwnershipRegistry {
    struct Record {
        address owner;
        bytes32 contentHash;   // keccak256(pHash ‖ metaHash)
        uint64  timestamp;
        bool    doNotTrain;    // AI 학습 거부 플래그(온체인 명시)
    }

    mapping(uint256 => Record) public records;      // tokenId => Record
    mapping(bytes32 => uint256) public hashToToken; // 중복 등록 방지
    uint256 public nextId;

    event Registered(uint256 indexed id, address indexed owner, bytes32 contentHash);
    event Transferred(uint256 indexed id, address indexed from, address indexed to);

    function register(bytes32 contentHash, bool doNotTrain) external returns (uint256 id) {
        require(hashToToken[contentHash] == 0, "already registered");
        id = ++nextId;
        records[id] = Record(msg.sender, contentHash, uint64(block.timestamp), doNotTrain);
        hashToToken[contentHash] = id;
        emit Registered(id, msg.sender, contentHash);
    }

    function verify(bytes32 contentHash) external view returns (bool exists, address owner, uint64 ts) {
        uint256 id = hashToToken[contentHash];
        if (id == 0) return (false, address(0), 0);
        Record storage r = records[id];
        return (true, r.owner, r.timestamp);
    }

    function transfer(uint256 id, address to) external {
        require(records[id].owner == msg.sender, "not owner");
        emit Transferred(id, msg.sender, to);
        records[id].owner = to;
    }
}
```

> ERC-721(NFT)로 만들 수도 있지만, 초기엔 **경량 커스텀 레지스트리**가 단순하고 가스도 싸다. "판매/유통"이 필요해지면 그때 ERC-721로 승격.

### 5-4. 등록/검증 흐름
- **등록**: 보호본 생성 완료 → `blockchain-svc`가 `register(contentHash, doNotTrain)` 호출 → tx_hash를 `ownership_records`에 저장.
- **가스비 대납(가스리스)**: 크리에이터에게 지갑/가스 부담을 주지 않기 위해 **플랫폼 릴레이어**가 대신 서명·전송(메타트랜잭션). 릴레이어 개인키는 **C KMS로 보관·서명**.
- **검증(누구나)**: 이미지의 pHash를 재계산 → `verify(contentHash)`로 온체인에 최초 등록자/시각 조회 → 소유권 증명서(PDF) 출력.

### 5-5. asset-service ↔ blockchain-svc 통합 지점

`blockchain-svc`는 이미지·업로드·DB를 전혀 모른다 — 해시와 지갑 주소만 받아 온체인에 등록/조회하는 순수 API다. **asset-service가 이 오케스트레이터 역할을 실제로 구현했다** (`apps/asset-service/`, `src/orchestration.ts`) — 아래 설명은 더 이상 설계일 뿐이 아니라 실제 동작 확인된 내용이다. 실제 API 계약과 해시 공식은 [apps/blockchain-svc/INTEGRATION.md](../apps/blockchain-svc/INTEGRATION.md)에 상세히 정리해뒀고, 여기서는 asset-service 쪽 책임만 요약한다.

**호출 시점**: §1-1 업로드 파이프라인의 "보호본 저장 완료" 직후, "블록체인 등록" 단계에서 asset-service가 protection-svc의 출력(perceptualHash, metadataHash)을 받아 `POST /assets/register`를 호출한다.

**asset-service가 책임지는 것** (실제 구현됨):
- `allowAiTraining` 값을 뒤집어 `doNotTrain` boolean으로 변환해 전달
- `POST /assets/register` 응답의 `txHash`/`blockNumber`를 `ownership_records` 테이블에 저장 (§4 데이터 모델의 축소판, `apps/asset-service/src/db/schema.ts`)
- **동기 호출 금지**: `POST /artworks`는 즉시 202를 반환하고, 실제 오케스트레이션은 백그라운드로 실행 (`apps/asset-service/README.md`의 상태 머신 참고). 지금은 in-process fire-and-forget 방식 — Phase 2+에서 Redis/BullMQ 큐로 격리 예정
- **409(중복 해시) 처리 — 실제 구현 + 테스트까지 완료**: 응답의 `contentHash`로 `GET /assets/verify/:contentHash` 재조회 → on-chain owner가 asset-service가 알고 있는 owner와 같으면 멱등 처리(재시도로 간주) → `PUBLISHED`, 다르면 실제 해시 충돌이므로 `FAILED` + 명확한 에러 메시지 (`apps/asset-service/test/orchestration.test.ts`로 mocked 테스트 검증)
- **재시도 정책**: 아직 미구현 — 현재는 실패 시 바로 `FAILED`로 전환. 지수 백오프 재시도는 Phase 2+ 과제로 남음

**protection-svc가 책임지는 것** (INTEGRATION.md 상세): `perceptualHash`(공개용 보호본 기준)와 `metadataHash` 계산까지만 하고, 온체인 호출은 절대 직접 하지 않는다 — protection-svc는 이미지 변환에만 집중하는 순수 서비스로 유지.

---

## 6. KMS / 보안 컴플라이언스 (준비된 C 서버 연동)

### 6-1. KMS 경계 정의
KMS(C 서버)가 담당하는 것 / 담당하지 않는 것을 명확히 나눈다.

**KMS가 하는 일:**
- 원본 이미지 암호화 키(DEK) 발급·래핑 (Envelope Encryption: KMS는 KEK만 보관, DEK는 암호문으로 DB 저장)
- 블록체인 릴레이어 지갑 **개인키 보관 및 서명** (개인키가 앱 서버 메모리에 노출되지 않게)
- 증거 패키지 / 감사 로그 **디지털 서명** (위·변조 방지)
- C2PA manifest 서명용 키 관리

**KMS가 하지 않는 일:**
- 대용량 이미지 자체를 KMS로 직접 암호화(X). → 앱이 DEK로 암호화, KMS는 DEK만 관리.

### 6-2. 연동 방식
```
[Node/Rust 서비스] ──gRPC/유닉스소켓──▶ [kms-adapter(래퍼)] ──▶ [C KMS 서버]
   요청: WrapKey / UnwrapKey / Sign / Verify
```
- `/infra/kms-adapter`에 얇은 어댑터를 두고, 각 서비스는 어댑터의 명확한 인터페이스(`Sign(payload) -> signature` 등)만 호출.
- 통신은 mTLS. KMS 서버는 내부망에만 노출(공개 금지).

### 6-3. 컴플라이언스 체크리스트 (수립 필요 항목)
- [ ] 원본 저장소 암호화(at-rest) + 접근 최소권한
- [ ] 전송 구간 TLS 1.2+ / 내부 서비스 간 mTLS
- [ ] 감사 로그 **immutable**(append-only) + 서명
- [ ] 개인정보(이메일·지갑주소) 처리방침, 파기 절차
- [ ] 관리자 위험 액션 승인 워크플로우(다운로드/테이크다운)
- [ ] 대량 다운로드·내부자 오남용 탐지 알림
- [ ] 비밀정보(.env, 키) → Git 커밋 금지, secret manager 사용
- [ ] 콘텐츠 신고/삭제(테이크다운) 대응 SLA 정의

---

## 7. 침해 대응 매뉴얼 (런북) — "법적 레거시" 목표의 실행판

침해 의심이 탐지되거나 신고가 접수되면 아래 절차를 자동/반자동으로 실행한다.

```
1. 탐지/신고 접수
     └ detection-svc가 발견 or 사용자가 Report 제출
2. 자동 증거 수집 (스냅샷 시점 고정)
     ├ 침해 의심 URL 스크린샷 + HTML/HTTP 헤더 아카이브
     ├ 발견 이미지 pHash ↔ 원본 pHash 유사도 점수
     ├ 워터마크 검출 결과
     └ 온체인 소유권 레코드(최초 등록 시각·주소) 조회
3. 증거 패키지 생성 (서명)
     └ PDF(사람용) + JSON(기계용), KMS 서명 첨부 → Evidence 저장소(WORM)
4. 권리자 알림
     └ 크리에이터 대시보드에 케이스 생성 + 이메일
5. 대응 옵션 안내 (플랫폼은 "자료 제공"까지)
     ├ 플랫폼 내부 콘텐츠 → 즉시 테이크다운
     ├ 외부 사이트 → DMCA/권리침해 통지 템플릿 자동 작성
     └ AI 데이터셋 수록 정황 → Do-Not-Train 근거 + 온체인 증명 첨부
6. 케이스 상태 추적
     └ OPEN → EVIDENCE_READY → NOTIFIED → RESOLVED/ESCALATED
```

> 주의: 우리는 **증거 생성·자료 제공**까지 한다. 법적 판단·소송은 사용자/변호사 몫임을 UI에 명시(면책).

---

## 8. 상세 로드맵 (5인 팀 기준)

전제: 파트타임/학기 프로젝트 가정. 풀타임이면 기간 절반으로 압축 가능.

### Phase 0 — 준비 (2주)
- 리포지토리/CI 셋업(monorepo, lint, CI), docker-compose 로컬 스택
- 기술 스파이크: Glaze/Nightshade 원리 학습, Polygon 테스트넷 지갑, C KMS 어댑터 PoC
- 데이터 모델·API 계약(shared-types) 초안 확정
- **산출물**: 로컬에서 빈 서비스들이 뜨고 서로 헬스체크 통과

### Phase 1 — MVP: "올리고 보고 소유권 남긴다" (6~8주)
- 인증(소셜 로그인) + 크리에이터 지갑(커스터디)
- 업로드 → 검증 → SHA256/pHash → **원본 KMS 암호화 저장**
- 보호 v0: **invisible watermark + 리사이즈/썸네일**(무거운 ML 전) + C2PA 기본
- 블록체인 **소유권 등록(테스트넷)** + 검증 페이지
- 갤러리/피드/작품 상세(최소 커뮤니티)
- Delivery Gateway: signed URL + robots.txt/X-Robots-Tag
- **목표**: "업로드하면 보호본으로 공개되고, 온체인에 소유권이 남는다"

### Phase 2 — AI 학습 방지 엔진 v1 (6주)
- Style Confusion Layer(Glaze 계열) 구현 + 강도 프리셋(L1~L3)
- 시각 품질 평가(원본 대비 SSIM/LPIPS) 자동 리포트
- Rust로 이미지 파이프라인 성능 이관(병목 구간)
- 크리에이터 대시보드: 보호 적용률·공개 범위 관리
- **목표**: "정책적으로 보호 강도를 고르고, 화풍 모방을 실측으로 방해"

### Phase 3 — 침해 탐지 & 증거 (6주)
- detection-svc: pHash 유사도 + 역이미지 검색 연동
- 증거 패키지(PDF+JSON) 자동 생성 + KMS 서명
- 신고(Report)·모더레이션 큐·테이크다운
- 침해 대응 런북 반자동화(§7)
- **목표**: "무단 사용을 찾아 증거를 만들고 대응 자료를 제공"

### Phase 4 — 고도화 (4주+)
- Concept Misalignment Layer(Nightshade 계열, L4) 실험적 도입
- 허니에셋/허니URL로 크롤러 유인·탐지
- 적응형 안티스크랩(대량 수집 패턴 학습)
- 온체인 소유권 이전/ERC-721 승격 검토, 메인넷 전환
- **목표**: "무단 학습 비용 극대화 + 실서비스 운영성"
- **스코핑 완료** (구현 전 설계 패스): [`PHASE4_SCOPING.md`](PHASE4_SCOPING.md) —
  네 항목 각각의 기술적 요구사항, 기존 구축물과의 관계, 현실적 타당성 평가,
  우선 구현 vs. 보류 권고. 요약: (1) Concept Misalignment는 스타일 혼란
  레이어와 다른 최적화 대상(CLIP 텍스트-이미지 정렬)이 필요하고, 개별
  창작자 단위로는 "모델을 붕괴시킨다"가 아니라 "이 이미지의 캡션-특징
  연관이 틀어진다" 정도가 정직한 주장선. (2) 허니에셋/URL은
  delivery-gateway에 이미 만든 크롤러 분류를 바로 확장 가능. (3) 적응형
  안티스크랩은 실제 트래픽 데이터 없이 신뢰성 있게 만들 수 없음 — 순차
  ID 열거 탐지만 지금 구현할 가치 있고 나머지는 실배포 후로 보류 권고.
  (4) 메인넷 전환은 보안 감사가 하드 블로커, ERC-721 승격은 별개 결정으로
  분리(지금은 비권장 — 이 프로젝트의 실제 요구사항엔 마켓플레이스 호환성이
  불필요, approve 기반 전송은 실제 탈취 벡터 1순위).
- **부분 구현 완료**: (2) 허니에셋/URL —
  `apps/delivery-gateway/src/honeypot.rs`. `/decoy/:token` 라우트가
  실제 페이지에서는 절대 링크되지 않고 `robots.txt`의 `Disallow`
  목록에만 등장하므로, 그 경로로 들어오는 요청은 구조적으로
  크롤러/스크레이퍼로 확정됨. 매번 유효한 200 OK 디코이 PNG를
  반환(403이 아님 — 스크레이퍼가 걸린 걸 눈치채고 행동을 바꾸지
  못하게). 원래 스코핑이 제안한 "실제 아트워크에 히트당 고유
  워터마크 페이로드를 심어 유출 시 역추적" 부분은 구현 범위를
  축소해 정적 디코이 이미지로 단순화 — 세부 사유는
  `PHASE4_SCOPING.md` §2 참고. (3) 적응형 안티스크랩 중 순차 ID
  열거 탐지도 완료 — `src/enumeration.rs`, 단 이 프로젝트의 실제
  아트워크 ID는 순차가 아닌 랜덤 UUID라 "서로 다른 아트워크
  접근 개수" 신호로 대체 구현(`PHASE4_SCOPING.md` §3).

### 마일스톤 요약
| 시점 | 데모 가능한 것 |
|---|---|
| ~10주 | 업로드·보호·온체인 등록·갤러리 (MVP) |
| ~16주 | 화풍 방해 엔진 + 강도 프리셋 |
| ~22주 | 침해 탐지 + 증거 패키지 |
| 22주+ | 고급 방어·운영 안정화 |

---

## 9. 팀 역할 분담 (5명)

| # | 역할 | 담당 서비스 | 주 언어 |
|---|---|---|---|
| 1 | **Frontend** | web (갤러리·업로드·대시보드·검증 페이지) | React/TS |
| 2 | **Backend Core** | api-gateway, asset/community-service, 인증 | Node.js |
| 3 | **AI/ML** | protection ml-engine, detection-svc | Python |
| 4 | **Systems** | protection rust-core, delivery-gateway | Rust |
| 5 | **Blockchain + Security** | blockchain-svc, contracts, KMS 연동, 컴플라이언스 | Solidity/Node + C 연동 |

> 5인은 서비스 수보다 적으므로 **Phase별로 집중 배치**한다. 예: Phase 1은 전원이 MVP(1·2·5번 중심), Phase 2는 3·4번이 엔진 집중, 2·1번이 대시보드. 협업 채널은 GitHub(코드/이슈/PR) · Discord(상시 개발) · KakaoTalk(빠른 공지).

---

## 10. 기술 스택 요약

```
Frontend    React, TypeScript, Vite, TanStack Query, Zustand, Tailwind
BE Core     Node.js (NestJS or Fastify), PostgreSQL, Redis(BullMQ)
Protection  Rust (axum, image crate) + Python (PyTorch, OpenCV, Pillow, libvips)
Delivery    Rust (axum + tower), NGINX/Envoy 앞단
Blockchain  Solidity (Foundry/Hardhat), ethers.js, Polygon(Amoy→Mainnet)
Provenance  C2PA 호환 라이브러리
KMS         준비된 C 서버 + gRPC/소켓 어댑터 (Envelope Encryption, Signing)
Storage     S3 호환 Object Store (원본/보호본/증거 버킷 분리, 증거는 WORM/버전)
Infra       Docker Compose(로컬) → k8s or Fly.io, GitHub Actions(CI)
Observability OpenTelemetry, Prometheus, Grafana, Loki
```

---

## 11. 우선순위 (지금 당장 만든다면 이 순서)

```
1. Asset/Community Registry + 업로드/해시
2. KMS 연동(원본 암호화) + 원본/보호본 저장소 분리
3. Blockchain 소유권 등록(테스트넷)
4. Delivery Gateway(signed URL) + robots 정책
5. Protection v1(워터마크 → Style Confusion)
6. Detection + Evidence Pack
7. 고급 Anti-Training(Concept Misalignment) / 허니에셋
```
이유: 화려한 "학습 방해 엔진"보다, **사용자가 실제로 쓸 수 있는 커뮤니티 + 소유권 증명 골격**을 먼저 세워야 프로젝트가 굴러간다.

---

## 12. 현실적 한계 (팀 전원 합의 필요)

1. **완전 차단 불가**: 스크린샷 → 리터칭 → 재라벨링 → 폐쇄 환경 학습을 하면 막을 수 없다. Glaze 개발팀도 영구 해결책이 아니라고 명시.
2. **robots는 협조형**: 존중하는 크롤러에만 유효. 악성 수집자는 게이트웨이 접근통제로 대응.
3. **anti-training은 트레이드오프**: 강할수록 감상/인쇄/압축 품질과 충돌. 그래서 **원본 무변형 보존 + 배포본만 보호 + 강도 옵트인**이 원칙.
4. **블록체인은 "선점 증명"이지 "저작권 그 자체"가 아니다**: 온체인 타임스탬프는 "이 시점에 이 사람이 이 해시를 등록했다"를 증명할 뿐, 법적 저작권 귀속과는 다르다. 마케팅에서 과장 금지.
5. 따라서 제품 메시지는 **"차단"이 아니라 "억제 · 추적 · 증빙 · 학습 경제성 파괴"**.

---

## 13. 다음에 만들면 좋은 산출물

- ~~`contracts/OwnershipRegistry.sol` 실제 구현 + 테스트(Foundry)~~ — 완료, Amoy 테스트넷 배포+검증까지 (`contracts/DEPLOYMENTS.md`)
- ~~protection ml-engine PoC (Glaze 유사 style-cloak 최소 구현)~~ — 완료, 실제 명화 검증 + EOT 견고성 실험까지 (`apps/protection-svc/ml-engine/README.md`)
- ~~perceptualHash 함수 구현~~ — 완료, `ml-engine/src/perceptual_hash.py` (표준 DCT pHash, 32바이트 정확히 매칭, Hamming distance로 검증까지)
- ~~rust-core 워터마크 PoC~~ — 완료, DCT 계수관계 기반 워터마크 + JPEG/리사이즈 견고성 실측, ML cloak와의 실패지점 비교까지 (`apps/protection-svc/rust-core/README.md`)
- ~~rust-core C2PA 매니페스트~~ — 완료. `c2pa` 크레이트(v0.89, `rust_native_crypto`) 연동해서 매니페스트 임베딩·커스텀 assertion·콘텐츠해시 무결성 검증까지 동작. 한때 **클레임 서명 자체가 재검증 시 통과하지 못하는 이슈**(`claimSignature.mismatch`)가 있었으나, 업스트림 `c2pa-rs`의 알려진 버그로 근본 원인 확인(`contentauth/c2pa-rs#2262`/`#2150` — 인증서에 Organization(O) 속성이 없으면 실제로는 유효한 서명이 서명 불일치로 오보고됨). 자체 서명 인증서에 `OrganizationName`을 추가하는 것으로 해결, 실제 CLI 왕복(`c2pa-sign`→`c2pa-verify`)으로 `validation_state: Valid` 확인까지 완료. 남은 건 자체서명 인증서라 `signingCredential.untrusted`(신뢰 목록 미등재)뿐 — 이건 암호학적 문제가 아니라 운영상 PKI 문제. 상세는 `apps/protection-svc/rust-core/README.md` 참고. **소유권 증명은 여전히 blockchain-svc가 1차 메커니즘이지만, C2PA 서명도 이제 "진짜 암호학적 보증"을 제공한다**
- ~~rust-core 해상도 변형본(썸네일) 생성 파이프라인~~ — 완료. `PROJECT_DESIGN.md` §3-5 변형본(2048/1280/그리드 썸네일) 생성 + 각 변형본에 "이 배율에서 보호가 실측으로 안전한지" 태그(`Safe`/`Unknown`/`Unsafe`, 0.5x/0.25x 실측 경계 기준)를 자동으로 붙임 (`apps/protection-svc/rust-core/README.md`)
- ~~ml-engine → rust-core → perceptualHash 호출까지 잇는 실제 오케스트레이션 코드~~ — 완료, `apps/protection-svc/orchestrate.py`. 반고흐 명화로 전체 파이프라인 실행 + blockchain-svc 온체인 등록까지 실제로 성공(§3-8 실측 항목 참고). **256px 처리 해상도 고정 문제를 이 과정에서 처음 발견**
- ~~256px 이상 해상도에서 프리셋/EOT/견고성 수치 재검증~~ — 완료, 1024px로 실측 (`ml-engine/README.md`, `rust-core/README.md`). **핵심 발견: ML cloak의 0.25x 리사이즈 붕괴는 해상도를 올리면 완화되지만(-144%→-14%, 정보이론적 한계라 절대 픽셀수가 늘면 개선), 워터마크의 0.25x 붕괴는 해상도를 올려도 거의 그대로(37.5%→29.7% BER, 기하학적 블록 어긋남이라 비율 문제일 뿐 절대 해상도와 무관)** — 두 메커니즘이 서로 다른 원인으로 실패한다는 가설을 실측으로 교차검증함. 부수적으로 GPU VRAM 용량계획 이슈도 발견(`eot_samples × size`가 곱으로 VRAM을 소모, 8GB GPU에서 1024px+samples=3 조합이 94.5% VRAM으로 심각한 지연 유발 → samples=1로 낮춰서 해결)
- ~~protection-svc를 실제 HTTP 서비스로 wrapping~~ — 완료, `apps/protection-svc/server.py`(FastAPI). `POST /protect`→`GET /protect/{jobId}` job API를 INTEGRATION.md 계약 그대로 구현, in-process 단일 워커(GPU VRAM 용량 이슈 때문에 의도적으로 동시 1개 제한). **실제 HTTP로 전체 루프 검증**: protection-svc에 명화 등록 → job 완료 폴링 → 그 결과(perceptualHash/metadataHash)를 blockchain-svc의 `POST /assets/register`에 실제 HTTP 호출 → 온체인 등록 → `GET /assets/verify`로 최종 확인까지 성공. 두 개의 독립된 서비스가 진짜 네트워크 호출로 이어진 것까지 증명함. 남은 한계: imageUri는 로컬 파일 경로일 뿐(오브젝트 스토리지 미연동), job 상태는 인메모리(재시작 시 유실), 인증 없음 — 전부 `server.py` docstring에 명시
- ~~asset-service 오케스트레이션 코드~~ — 완료, `apps/asset-service/`(Node/TS/Express, SQLite+Drizzle). `POST /artworks`→`GET /artworks/:id` 업로드 상태 머신(`UPLOADED→PROTECTING→REGISTERING→PUBLISHED`/`FAILED`) 구현. **세 개의 독립 HTTP 서비스(asset-service, protection-svc, blockchain-svc)가 실제 네트워크 호출로 이어져서 명화 업로드→온체인 등록까지 끝까지 검증됨** (`apps/asset-service/README.md`). blockchain-svc/INTEGRATION.md에 설계만 해뒀던 409(중복 해시) 멱등/충돌 처리도 이번에 실제 구현 + mocked 테스트 4종으로 검증.
- ~~asset-service의 §3-2 커뮤니티 기능(작품 CRUD, 피드, 팔로우/좋아요, 신고/모더레이션)~~ — 완료. `src/routes/community.ts`: 피드(`latest`/`popular`/`following`), 팔로우, 좋아요(멱등), 북마크/컬렉션, 댓글, 신고→모더레이션 큐(`PENDING→RESOLVED`/`DISMISSED`, 이미 처리된 신고 재처리는 409). api-gateway 쪽에 동일 패턴의 프록시 라우터 추가(`apps/api-gateway/src/routes/community.ts`) — JWT에서 신원 주입(요청 바디의 userId는 무시), 모더레이션 엔드포인트는 `MODERATOR`/`ADMIN` 역할로 게이팅. 로컬에서 실제 HTTP로 전체 체인(회원가입→JWT→좋아요 프록시→asset-service 실제 반영) E2E 검증까지 완료. `visibility=followers`는 저장은 되지만 강제되지 않음(이 서비스가 조회자 신원을 모름 — 알려진 한계, README에 명시).
- ~~Delivery Gateway(§3-5, Rust)~~ — 완료, `apps/delivery-gateway/`(axum). 정책 기반 signed URL(`POST /internal/sign` → `GET /asset/:id/render?variant=...&exp=...&sig=...`, HMAC-SHA256로 `(artworkId, variant)` 쌍 + 짧은 TTL에 바인딩 — 다른 variant나 artwork로 재사용 불가), 뷰어별 변형본 선택(비로그인→1280px, 로그인→2048px, rust-core의 기존 grid_thumbnail_512도 노출), 매 렌더 요청마다 실제 접근 통제(서명/만료 검증 → AI 크롤러 UA 차단(GPTBot/ClaudeBot/Google-Extended 등) → referer 화이트리스트(hotlink 차단, Referer 없으면 통과) → IP별 rate limit → asset-service에 실제 HTTP로 `assetVersions` 조회 후 파일 서빙, `X-Robots-Tag: noindex, noimageindex` 포함), `robots.txt` 자동 생성(크롤러 차단 목록과 동일 소스 공유 — 협조형 신호일 뿐 실제 방어는 렌더 핸들러가 담당). 유닛 테스트 11개 + `wiremock`으로 실제 asset-service를 흉내낸 통합 테스트 10개, 로컬에서 실제 asset-service 인스턴스 대상 진짜 E2E(서명 발급→실제 HTTP 조회→실제 파일 바이트 서빙→크롤러/변조 차단)까지 검증. decoy/honeypot 응답은 Phase 4 범위로 남김, rate limiter는 인메모리(단일 인스턴스 한정, 실제 배포는 Redis 필요) — `apps/delivery-gateway/README.md`에 명시.
- ~~Phase 4 스코핑~~ — 완료, [`PHASE4_SCOPING.md`](PHASE4_SCOPING.md). §8 Phase 4의 네 항목(Concept Misalignment Layer, 허니에셋/URL, 적응형 안티스크랩, 메인넷 전환/ERC-721) 각각의 기술 요구사항·기존 구축물과의 관계·현실적 타당성·우선순위 권고. 상세 요약은 이 문서의 "Phase 4 — 고도화" 절 참고.
- ~~적응형 안티스크랩(스코핑에서 권고한 첫 항목)~~ — 완료, `apps/delivery-gateway/src/enumeration.rs`. 원래 스코핑 문서는 "순차 ID 열거 탐지"를 제안했으나, 실제 구현 과정에서 이 프로젝트의 작품 ID가 `randomUUID` 기반 16자 랜덤 hex 문자열(`ast_...`)이라 애초에 순차 열거가 불가능하다는 걸 확인 — 같은 근본 신호(정상 사용자는 세션당 소수의 작품만 조회, 스크레이퍼는 짧은 시간에 다수의 *서로 다른* 작품을 조회)를 ID 체계에 맞게 적용해 "슬라이딩 윈도우 내 IP당 서로 다른 작품 조회 개수" 탐지로 구현. 같은 작품 반복 조회(실제 새로고침)는 절대 걸리지 않음. IP 기반이라 IP 로테이션엔 무력하다는 한계는 명시 — 스코핑 문서가 애초에 "지금 만들 가치 있는 저비용 신호"로만 권고했던 항목이고, 전체 행동기반 평판 시스템은 여전히 실배포 트래픽 없이는 보류.
- ~~OpenAPI 스펙(api-gateway) 초안~~ — 완료, `apps/api-gateway/openapi.yaml`
- ~~`docker-compose.yml`로 전체 로컬 스택 부팅~~ — 완료, api-gateway/web/delivery-gateway까지 전부 포함
- ~~KMS 어댑터 인터페이스 정의(proto)~~ — 완료, `infra/kms-adapter`(실제 C 서버 프로토콜을 미러링, wrapKey/unwrapKey 실전 검증까지)

> 위 중 어떤 것부터 실제 코드로 만들지 알려주면 이어서 구현해줄게.
