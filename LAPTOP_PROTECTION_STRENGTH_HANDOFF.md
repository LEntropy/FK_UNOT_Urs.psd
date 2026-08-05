# 개인 노트북용 Protection Strength Test 인수인계

## 1. 목적

목표는 **원본 이미지가 사람 눈에 이질적으로 보이지 않는 범위에서 가장 강한 보호 강도**를 찾는 것이다.

이 실험은 팀의 `apps/protection-svc`에 이미 있는 Style Cloak, watermark, C2PA를 다시 만드는 목적이 아니다.
우리가 추가로 검증하려는 핵심은 다음이다.

> 팀 서비스의 기존 L2 보호 기준 위에 P2T(Perceptual-Preserving Transform)를 추가하거나,
> 보호 강도를 조정했을 때 시각 품질을 유지하면서 학습 방어 가능성을 높일 수 있는가?

LoRA 공격 검증은 최종 후보만 Google Colab GPU에서 한다. 개인 노트북에서는 우선 5장의
대표 이미지로 보호본의 시각 품질과 강도 한계를 탐색한다.

## 2. 전달된 파일 구성

이 문서와 함께 전달된 폴더/파일:

```text
protection_pipeline/        # P2T, SCL, CML, TL 코드 및 프리셋
calibration-samples/        # 대표 이미지 5장
art/metadata.jsonl          # 이미지별 캡션
requirements.txt
```

대표 이미지:

```text
art2.jpg, art7.jpg, art8.jpg, art16.jpg, art24.jpg
```

`art2`, `art7`, `art8`은 기존 강한 L3에서 변형이 크게 나타난 이미지라 품질 확인 대상으로
선정했다.

## 3. 개인 노트북 환경 설정

Windows 기준으로 WSL Ubuntu + VS Code 사용을 권장한다. 기존 `.venv`는 복사하지 말고,
개인 노트북에서 새로 만든다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy Pillow torch torchvision transformers
```

GPU 확인:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음')"
```

- CUDA가 `True`면 명령에 `--device cuda` 사용
- CUDA가 `False`면 `--device cpu` 사용. 5장은 가능하지만 SCL/CML은 느릴 수 있음

## 4. 실행 명령 형식

모든 보호본 생성은 아래 모듈을 사용한다.

```bash
python -m protection_pipeline.run_integrated_protection \
  --input-dir calibration-samples \
  --captions art/metadata.jsonl \
  --output-dir <출력_폴더명> \
  --profile <프리셋명> \
  --seed 42 \
  --device cuda
```

seed `42`는 결과 재현을 위한 고정값이다. 실제 배포에는 작품별 고유 seed를 사용한다.

## 5. 기준값: 기존 L2

`L2_PORTFOLIO`는 사용자가 원본과 비교했을 때 이질감이 없다고 판단한 기준값이다.
이후 탐색은 L2보다 약한 값이 아니라 **L2보다 조금 강한 값**에서 시작한다.

| 계층 | L2 값 |
| --- | ---: |
| P2T phase | 0.035 |
| P2T residual | 0.035 |
| P2T local contrast | 0.018 |
| P2T edge | 0.012 |
| SCL steps | 8 |
| SCL epsilon | 0.015 |
| CML steps | 8 |
| CML epsilon | 0.015 |

## 6. 현재 진행 상태

완료:

- P2T만 / P2T+약한 SCL / P2T+약한 SCL+CML의 매우 약한 교정값 테스트
- L2 기준 확인
- `L2_PLUS_STEP_1` 대표 5장 생성 및 비교
- `L2_PLUS_STEP_2` 대표 5장 생성 및 비교

기존 강한 후보는 탈락:

- `L3_ANTI_TRAIN`: 넓은 영역에 노이즈/지저분함이 보임
- `L3_BALANCED_CANDIDATE`: 사용자 육안 기준에서 이질감이 보임

중요: 숫자상 PSNR/MAE가 좋아도 사용자 눈에 이질감이 보이면 해당 후보는 탈락이다.

## 7. 현재 가장 강한 생성 완료 후보

### L2+ 1단계

| 계층 | 설정값 | L2 대비 |
| --- | ---: | ---: |
| P2T phase | 0.040 | +0.005 |
| P2T residual | 0.040 | +0.005 |
| P2T local contrast | 0.020 | +0.002 |
| P2T edge | 0.014 | +0.002 |
| SCL steps | 10 | +2 |
| SCL epsilon | 0.017 | +0.002 |
| CML steps | 10 | +2 |
| CML epsilon | 0.017 | +0.002 |

### L2+ 2단계

| 계층 | 설정값 | L2 대비 |
| --- | ---: | ---: |
| P2T phase | 0.042 | +0.007 |
| P2T residual | 0.042 | +0.007 |
| P2T local contrast | 0.021 | +0.003 |
| P2T edge | 0.015 | +0.003 |
| SCL steps | 11 | +3 |
| SCL epsilon | 0.018 | +0.003 |
| CML steps | 11 | +3 |
| CML epsilon | 0.018 | +0.003 |

생성 명령:

```bash
python -m protection_pipeline.run_integrated_protection --input-dir calibration-samples --captions art/metadata.jsonl --output-dir calibration-l2-plus-2 --profile L2_PLUS_STEP_2 --seed 42 --device cuda
```

## 8. 다음 작업

1. `calibration-l2-plus-2`를 원본/L2/L2+1과 비교한다.
2. 육안 이질감이 없으면, L2+3을 **아주 작은 폭**으로 만든다.
3. 이질감이 생기면 바로 이전 단계를 최대 허용 강도 후보로 고정한다.
4. 최대 후보를 전체 `art/` 데이터셋에 적용한다.
5. 최종 후보만 Colab에서 LoRA 공격(원본 대비 보호본)으로 검증한다.

프리셋 표를 제시할 때는 항상 L2 대비 `+/-` 값도 함께 표시한다.

## 9. 팀 `protection-svc`와의 관계

팀 서비스의 Style Cloak 프리셋(`epsilon`, `steps`, `color_weight`, `mask_low`,
`clip_transfer_weight`, EOT)은 이 프로젝트의 P2T/SCL/CML 수치와 1:1 변환되지 않는다.
알고리즘과 손실함수가 다르기 때문이다.

따라서 최종 목표는 팀 서비스의 기존 보호 기능 위에 P2T 같은 추가 계층을 붙였을 때,
육안 품질을 유지하면서 방어 효과가 더 좋아지는지 확인하는 것이다.

팀 서비스 문서에서 확인된 주의 사항:

- 강한 L3는 색감/평탄 영역 노이즈 문제가 있었음
- CML은 팀의 SD1.5 LoRA 검증에서 WEAK/FAIL이어서 기본 비활성
- watermark는 0.25배 축소에서 약해짐
- C2PA는 개발용 자체서명이며 운영에는 KMS/정식 인증서 필요

## 10. Codex에게 처음 전달할 문장

개인 노트북의 Codex에서 이 파일을 연 뒤 아래처럼 요청한다.

> `LAPTOP_PROTECTION_STRENGTH_HANDOFF.md를 읽고 보호강도 테스트를 이어가자. 현재 L2+2 대표 5장까지 생성했으며, 육안 이질감이 없는 최대 강도를 찾는 것이 목표다. 다음 단계는 원본/L2/L2+1/L2+2 비교를 바탕으로 L2+3을 만들지 결정하는 것이다.`
