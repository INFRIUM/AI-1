# AI-1
INFREUM AI1 - 센서 데이터 전처리, 특징값 추출, Isolation Forest을 사용한 이상치 탐지

## 하는 일

교량 가속도 데이터를 Window 단위로 나눠 특징값을 계산하고, 정상 데이터로 학습한 Isolation Forest 로
Window 마다 이상 점수와 정상/이상 판정을 낸다. 결과는 AI②(위험 등급)에 파일 또는 API 로 전달한다.

- HW 축소 모형 데이터: 타격 1번 = Window 1개, 3축 가속도(LIS2DW12) 2개 + 온도
- 공개 교량 데이터 (AI①·AI② 합의서 v3, INFRIUM/AI-1#3): Ponte Moesa(박스교), HBTA(트러스교), Old ADA(트러스교, 점검용)
  - 원본과 HW 가속도계 수준 열화본(LIS2DW12) 두 벌로 처리
  - AI① v1(진동 크기 특징)과 v2(모양 특징, 참고 결과) 결과를 함께 전달

## 폴더

| 경로 | 내용 |
|---|---|
| `src/config.py` | 모든 설정값 (HW 규격, 교량·데이터셋, Window, 판정 기준). 숫자는 여기서만 바꾼다 |
| `src/convert_public.py` | 공개 데이터 → AI① 데이터 형식 변환 + LIS2DW12 열화본 생성 |
| `src/preprocess.py` | 전처리, Window 생성, 특징 추출, 판정 보류, 1-TAC·AR(AI② 지표) |
| `src/train.py` | 정상 기준·Isolation Forest 학습, v2 특징 선택 (`--select-v2`) |
| `src/predict.py` | 이상 점수·판정 |
| `src/evaluate.py` | 평가 (HW: 혼동행렬 등, 공개 데이터: 손상 단계별 판정 수) |
| `src/export_for_ai2.py` | AI② 전달 파일 (`public` 인자: 공개 데이터 전달 파일 10개) |
| `src/ai2_message.py`, `src/ai1_server.py` | 실시간 전송 메시지, 수신 서버 (HW 단계) |
| `data/original/README.md` | 공개 데이터 내려받을 파일 목록과 출처 |
| `results/public/for_AI2/` | AI② 전달 파일 (`AI1-public-1.1`) |
| `results/public/v2_selection/` | v2 특징 선택 표 (정상 데이터만 사용) |

원본·변환 데이터(`data/`), 모델(`models/`), 중간 결과는 용량이 커서 저장소에 올리지 않는다.

## 실행 (PowerShell, 프로젝트 폴더에서)

```powershell
pip install -r requirements.txt

# 1. 공개 데이터 변환 (data/original/README.md 의 파일을 먼저 내려받아 둔다)
python src/convert_public.py                      # hbta / ponte_moesa / old_ada

# 2~5. 데이터셋·버전별 전처리 → 학습 → 판정 → 평가
$env:IOT_BRIDGE = "hbta"; $env:IOT_SOURCE = "public_raw"   # 열화본은 public_deg (원본을 먼저 실행)
python src/preprocess.py
python src/train.py
python src/predict.py
python src/evaluate.py
python src/train.py --select-v2                   # v2 특징 선택 (원본·열화본 전처리 후)
$env:IOT_MODEL = "v2"                             # v2 학습·판정·평가는 같은 명령을 다시 실행

# 6. AI② 전달 파일
python src/export_for_ai2.py public
```

HW 축소 모형 데이터는 `IOT_SOURCE` 를 설정하지 않고(`real`) `python src/run_pipeline.py` 로 실행한다.
