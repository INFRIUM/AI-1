"""
프로젝트 설정 파일.

하드웨어가 확정되거나 바뀌면 **이 파일의 숫자와 이름만** 고친다. 다른 코드는 수정하지 않는다.
현재 기준: HW 센서 구성 변경 합의 (2026-10-12)
  - 센서: 3축 가속도(LIS2DW12) 교량당 2개 + 하중(공용) + 온도(교량당 1개, 주변 공기)
  - 변형률·변위 센서 제외
  - Ground Truth: D1 연결부 풀림 / D2 지지조건 변경 / D3 부재 강성 변경 (두 교량 공통)
  - 편심하중은 이상이 아니라 '실험 조건' (하중 크기 × 위치)

[A] 설정 항목
    - BRIDGES    : 교량 정보 + 센서 등록 정보 (ID, 위치, 측정 속도, 단위, 축) + 이상상태(D1~D3) 구현 방식
    - EXPERIMENT : 실험 조건 (하중 단계, 하중 위치, 계측 시간)
    - GROUND_TRUTH, QUALITY : 정답 코드 체계, 데이터 품질 플래그

재질: 두 교량 모두 알루미늄 축소모형 (E ≈ 69 GPa 로 강재의 약 1/3, 열팽창 23 µε/°C 로 강재의 약 2배).
      bridge_type(BOX/TRUSS)은 '구조형식', model_material(ALUMINUM)은 '모형 재료'로 따로 기록한다.
      결과는 실제 강교량의 절대 안전도가 아니라 모형에서 재현한 정상 대비 상대적인 차이로 해석한다.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 데이터 폴더 이름 (data/raw/<DATA_SOURCE>/<bridge>/ 에 실제 센서 데이터를 둔다)
#   real       : HW 축소모형 센서 데이터
#   public_raw : 공개 데이터 원본을 변환한 것 (convert_public.py)
#   public_deg : 공개 데이터를 HW 가속도계(LIS2DW12) 수준으로 열화한 것 (convert_public.py)
# 파일을 고치지 않고 바꾸려면 PowerShell 에서:  $env:IOT_SOURCE = "public_deg"
DATA_SOURCE = os.environ.get("IOT_SOURCE", "real")
# AI① 모델 버전 (공개 데이터). v1 = 진동 크기 Feature, v2 = 모양 Feature (AI② 합의, 결과 열람 후 변경한 참고 방법)
# 실행: $env:IOT_MODEL = "v2". v2 의 모델·결과 파일 이름에는 _v2 가 붙는다
AI1_MODEL = os.environ.get("IOT_MODEL", "v1")
MODEL_SUFFIX = "" if AI1_MODEL == "v1" else f"_{AI1_MODEL}"
RAW_DIR = ROOT / "data" / "raw" / DATA_SOURCE
PROCESSED_DIR = ROOT / "data" / "processed" / DATA_SOURCE
MODELS_DIR = ROOT / "models" / DATA_SOURCE
RESULTS_DIR = ROOT / "results" / DATA_SOURCE

# 현재 작업 대상 교량 (BRIDGES 의 키). 기본은 "box".
# 파일을 고치지 않고 바꾸려면 PowerShell 에서:  $env:IOT_BRIDGE = "truss"
# 두 교량을 한 번에 실행하려면:                python src/run_pipeline.py
ACTIVE_BRIDGE = os.environ.get("IOT_BRIDGE", "box")

GRAVITY = 9.80665  # [m/s^2]. 하중 kg → N 변환, 수직 가속도계의 중력 성분

# 센서 종류. 가속도 = 구조 응답 (모델 입력), 하중·온도 = 실험 조건 (모델 입력 아님)
SENSOR_TYPES = ("acceleration", "load", "temperature")

# 센서 원본 데이터의 열 (AI① 데이터 사양). HW 가 보내는 실제 데이터도 이 형식을 따른다.
# 3축 센서는 축마다 행이 따로 있다 (axis = X / Y / Z). 1축 센서는 axis = "NA"
RAW_COLUMNS = [
    "experiment_id", "bridge_type", "damage_type", "severity_level", "sensor_id",
    "timestamp_us", "axis", "value", "unit", "sample_rate_hz", "quality_flag",
]

# =============================================================================
# [A-1] 교량 / 센서 등록 정보  ← HW 확정 시 수정
# =============================================================================
# 공통: 경간 L = 1200 mm, 상판 폭 250 mm, 재료 ALUMINUM
# x_ratio        : 교량 길이방향 위치. 좌측 받침 0.0 ~ L/4 0.25 (300mm) ~ L/2 0.5 (600mm) ~ 우측 받침 1.0
# y_mm           : 교량 폭방향 위치 (중심선 0). 측면 센서는 비틀림 응답 확인용
# sample_rate_hz : 센서가 1초에 측정하는 횟수. 가속도 400 Hz 는 개발용 값 (HW 가 400~800 Hz 시험 후 확정)
# axes           : 측정 축. 3축 가속도 = X(길이방향) / Y(폭방향) / Z(수직), 1축 센서 = ["NA"]
# 가정 (HW 확인 필요): ACC_01 = L/4 중심선, ACC_02 = L/2 측면(+y). 온도 = 교량 주변 공기 온도
SPAN_MM = 1200
ACC_AXES = ["X", "Y", "Z"]

BRIDGES = {
    "box": {
        "bridge_id": "BOX_01",
        "bridge_type": "BOX",                  # 구조형식 코드 (데이터 사양)
        "structure_form": "STEEL_BOX_GIRDER",  # 본뜬 실제 구조형식
        "model_material": "ALUMINUM",          # 축소모형 제작 재료 (강재 아님)
        "exp_prefix": "BOX",
        "girder_line_y_mm": 100,               # 좌우 거더 라인의 중심선으로부터 거리 (가정)
        "sensors": [
            {"sensor_id": "BOX_ACC_01",  "sensor_type": "acceleration", "device": "LIS2DW12",          "sensor_location": "span_L/4",            "x_ratio": 0.25, "y_mm": 0,   "sample_rate_hz": 400, "unit": "m/s^2", "axes": ACC_AXES},
            {"sensor_id": "BOX_ACC_02",  "sensor_type": "acceleration", "device": "LIS2DW12",          "sensor_location": "span_L/2_side",       "x_ratio": 0.50, "y_mm": 100, "sample_rate_hz": 400, "unit": "m/s^2", "axes": ACC_AXES},
            {"sensor_id": "LOAD_01",     "sensor_type": "load",         "device": "TAL220 5kg+HX711",  "sensor_location": "load_point",          "x_ratio": 0.50, "y_mm": 0,   "sample_rate_hz": 80,  "unit": "N",     "axes": ["NA"]},
            {"sensor_id": "BOX_TEMP_01", "sensor_type": "temperature",  "device": "미정",              "sensor_location": "ambient_air",         "x_ratio": 0.50, "y_mm": 0,   "sample_rate_hz": 0.1, "unit": "°C",    "axes": ["NA"]},
        ],
        # 두 센서를 비교하는 특징: (이름, 분자 (센서, 축), 분모 (센서, 축)) → 두 신호 RMS 의 비
        "cross_features": [
            ("amp_ratio_L4_L2_z", ("BOX_ACC_01", "Z"), ("BOX_ACC_02", "Z")),   # 진동 형상 변화 (부재 강성·지지조건)
            ("torsion_y_z_L2", ("BOX_ACC_02", "Y"), ("BOX_ACC_02", "Z")),      # 측면 센서의 폭방향/수직 비 (비틀림)
        ],
        # 이상상태 구현 방식 (HW 합의 기준, 위치는 가정 → HW 확정 시 수정). 결과 문서 표기에 사용
        "damage_types": {
            "D1": {"name": "횡연결재 볼트 풀림",         "location": "crossbeam_L/4"},         # 탈착식 횡연결재 볼트 풀림 (HW 검토 중)
            "D2": {"name": "Roller 지점조건 변화",       "location": "right_roller_support"},  # 가정: 우측 받침이 Roller
            "D3": {"name": "x≈500 교체형 국부강성 변화", "location": "girder_x500"},           # 교체형 국부강성 (낮음 / 중간 / 큼)
        },
    },
    # Truss: 4 Panel × 300 mm (절점 x = 0, 300, 600, 900, 1200 mm).
    "truss": {
        "bridge_id": "TRUSS_01",
        "bridge_type": "TRUSS",
        "structure_form": "STEEL_TRUSS",
        "model_material": "ALUMINUM",
        "exp_prefix": "TRUSS",
        "girder_line_y_mm": 115,               # 좌우 트러스 면의 중심선으로부터 거리 (가정)
        "sensors": [
            {"sensor_id": "TRUSS_ACC_01",  "sensor_type": "acceleration", "device": "LIS2DW12",         "sensor_location": "joint_L/4",           "x_ratio": 0.25, "y_mm": 0,   "sample_rate_hz": 400, "unit": "m/s^2", "axes": ACC_AXES},
            {"sensor_id": "TRUSS_ACC_02",  "sensor_type": "acceleration", "device": "LIS2DW12",         "sensor_location": "joint_L/2_side",      "x_ratio": 0.50, "y_mm": 115, "sample_rate_hz": 400, "unit": "m/s^2", "axes": ACC_AXES},
            {"sensor_id": "LOAD_01",       "sensor_type": "load",         "device": "TAL220 5kg+HX711", "sensor_location": "load_point",          "x_ratio": 0.50, "y_mm": 0,   "sample_rate_hz": 80,  "unit": "N",     "axes": ["NA"]},
            {"sensor_id": "TRUSS_TEMP_01", "sensor_type": "temperature",  "device": "미정",             "sensor_location": "ambient_air",         "x_ratio": 0.50, "y_mm": 0,   "sample_rate_hz": 0.1, "unit": "°C",    "axes": ["NA"]},
        ],
        "cross_features": [
            ("amp_ratio_L4_L2_z", ("TRUSS_ACC_01", "Z"), ("TRUSS_ACC_02", "Z")),
            ("torsion_y_z_L2", ("TRUSS_ACC_02", "Y"), ("TRUSS_ACC_02", "Z")),
        ],
        "damage_types": {
            "D1": {"name": "M3 Test Joint 체결조건", "location": "test_joint_L/4"},    # 1/4 → 1/2 → 3/4 turn. 가정: L/4 절점
            "D2": {"name": "받침 조건 변경",         "location": "left_support"},      # HW 검토 중. 가정: 좌측 받침
            "D3": {"name": "교체형 대각재 강성저하", "location": "diagonal_center"},   # 가정: 중앙 패널 대각재 (x≈450 mm)
        },
    },
}


# -----------------------------------------------------------------------------
# 공개 데이터 교량 (합의서 v3). IOT_SOURCE = public_raw / public_deg 와 함께 쓴다.
# "public" 항목이 있는 교량은 preprocess 가 공개 데이터 규칙(실험 기록의 조건, 데이터셋별 Window·필터, 판정 보류,
# 1-TAC, AR)을 쓴다. 1-TAC·AR 은 AI② 위험 등급 지표이며 AI① 이상탐지 모델 입력에는 넣지 않는다.
# -----------------------------------------------------------------------------
def _acc(sid, device, location, x_ratio, y_mm, fs, axes, unit="m/s^2"):
    return {"sensor_id": sid, "sensor_type": "acceleration", "device": device, "sensor_location": location,
            "x_ratio": x_ratio, "y_mm": y_mm, "sample_rate_hz": fs, "unit": unit, "axes": axes}


BRIDGES.update({
    "ponte_moesa": {
        "bridge_id": "PONTE_MOESA", "bridge_type": "BOX", "structure_form": "PSC_BOX_GIRDER",
        "model_material": "PRESTRESSED_CONCRETE", "exp_prefix": "MOESA", "girder_line_y_mm": 3000,
        "sensors": [
            _acc("MOESA_ACC_01", "MR3000C (4번, S5)", "span2_3L/4", 0.75, 3000, 200, ACC_AXES),
            _acc("MOESA_ACC_02", "MR3000C (2번, 고정)", "span2_L/2", 0.50, -3000, 200, ACC_AXES),
            _acc("MOESA_ACC_03", "MR3000C (3번, 고정)", "span3_0.57L", 0.57, 3000, 200, ACC_AXES),
        ],
        "cross_features": [
            ("amp_ratio_L4_L2_z", ("MOESA_ACC_01", "Z"), ("MOESA_ACC_02", "Z")),
            ("torsion_y_z_L2", ("MOESA_ACC_02", "Y"), ("MOESA_ACC_02", "Z")),
        ],
        "damage_types": {},
        "public": {
            "window": {"mode": "fixed", "length_s": 60.0, "overlap": 0.5},
            "filter_band_hz": (2.0, 25.0),
            "acc_bands_hz": [(2, 8), (8, 15), (15, 25)],
            "use_damping": False,
            "pairs": {"4_2": ("MOESA_ACC_01", "MOESA_ACC_02"), "2_3": ("MOESA_ACC_02", "MOESA_ACC_03")},
            "main_pair": "4_2", "main_axis": "Z",
            "model_sensors": ("MOESA_ACC_01", "MOESA_ACC_02"),       # AI① 모델 입력 센서 (주 지표 쌍)
            "tac": {"pairs": [("4_2", "Z"), ("2_3", "Z")], "band_hz": (3.0, 12.0), "windows": 1, "nperseg": 2048},
            "ar": None,
            "raw_noise_ug_per_rthz": 10.0,    # MR3000C 사양 상한 → 원본 판정 보류 기준 (합의서 v3 7절)
            "ai1_model": "iforest",
        },
    },
    "hbta": {
        "bridge_id": "HBTA", "bridge_type": "TRUSS", "structure_form": "RIVETED_STEEL_TRUSS",
        "model_material": "STEEL", "exp_prefix": "HBTA", "girder_line_y_mm": 2250,
        "sensors": [
            _acc("HBTA_ACC_01", "Dytran (AG03)", "north_wall_x-7m", 0.30, 2250, 100, ["Y", "Z"]),
            _acc("HBTA_ACC_02", "Dytran (AG05)", "north_wall_midspan", 0.50, 2250, 100, ["Y", "Z"]),
        ],
        "cross_features": [
            ("amp_ratio_L4_L2_z", ("HBTA_ACC_01", "Z"), ("HBTA_ACC_02", "Z")),
            ("torsion_y_z_L2", ("HBTA_ACC_02", "Y"), ("HBTA_ACC_02", "Z")),
        ],
        "damage_types": {},
        "public": {
            "window": {"mode": "fixed", "length_s": 10.24, "overlap": 0.0},
            "filter_band_hz": (2.0, 25.0),
            "acc_bands_hz": [(2, 8), (8, 15), (15, 25)],
            "use_damping": False,
            "pairs": {"a_b": ("HBTA_ACC_01", "HBTA_ACC_02")},
            "main_pair": "a_b", "main_axis": "Z",
            "model_sensors": ("HBTA_ACC_01", "HBTA_ACC_02"),
            # 1-TAC 는 기본 Window 3개(30.72초)를 이어 붙여 계산 (합의 30초). 같은 값을 3개 Window 에 함께 기록
            "tac": {"pairs": [("a_b", "Z"), ("a_b", "Y")], "band_hz": (3.0, 25.0), "windows": 3, "nperseg": 1024},
            "ar": {"order": 5, "channels": [("HBTA_ACC_01", "Y"), ("HBTA_ACC_01", "Z"), ("HBTA_ACC_02", "Y"), ("HBTA_ACC_02", "Z")]},
            "raw_noise_ug_per_rthz": None,
            "ai1_model": "iforest",
        },
    },
    "old_ada": {
        "bridge_id": "OLD_ADA", "bridge_type": "TRUSS", "structure_form": "STEEL_WARREN_TRUSS",
        "model_material": "STEEL", "exp_prefix": "OLDADA", "girder_line_y_mm": 0,
        "sensors": [
            _acc("OLDADA_ACC_01", "ARS-A (A1)", "L/4", 0.25, 0, 200, ["Z"], unit="m/s^2 (추정)"),
            _acc("OLDADA_ACC_02", "ARS-A (A3)", "L/2", 0.50, 0, 200, ["Z"], unit="m/s^2 (추정)"),
        ],
        "cross_features": [("amp_ratio_L4_L2_z", ("OLDADA_ACC_01", "Z"), ("OLDADA_ACC_02", "Z"))],
        "damage_types": {},
        "public": {
            "window": {"mode": "fixed", "length_s": 30.0, "overlap": 0.0},   # 실험 = 자유 감쇠 30초 → Window 1개
            "filter_band_hz": (2.0, 25.0),
            "acc_bands_hz": [(2, 8), (8, 15), (15, 25)],
            "use_damping": True,                                             # 자유 감쇠라 감쇠비 계산 가능
            "pairs": {"a_b": ("OLDADA_ACC_01", "OLDADA_ACC_02")},
            "main_pair": "a_b", "main_axis": "Z",
            "model_sensors": ("OLDADA_ACC_01", "OLDADA_ACC_02"),
            "tac": {"pairs": [("a_b", "Z")], "band_hz": (2.0, 15.0), "windows": 1, "nperseg": 2048},
            "ar": {"order": 5, "channels": [("OLDADA_ACC_01", "Z"), ("OLDADA_ACC_02", "Z")]},
            "raw_noise_ug_per_rthz": None,
            # 정상 학습 주행이 4개뿐이라 Isolation Forest 대신 정상 대비 z-score 만 계산 (판정 없음, 서술 보고용)
            "ai1_model": "zscore",
        },
    },
})

# =============================================================================
# [A-2] 실험 조건  ← 실험 계획 확정 시 수정
# =============================================================================
# 하중은 '크기 × 위치' 조합이 실험 조건이다. D0 정상도, D1~D3 이상도 같은 조합으로 측정한다.
# 하중 0 kg 은 위치 구분이 없다.  조건 이름 예: "0kg", "1kg_C", "2kg_E"
EXPERIMENT = {
    "loads_kg": [0.0, 1.0, 2.0],   # 올리는 추 단계 [kg]. 로드셀은 N 으로 기록
    "load_positions": {            # 하중 위치 코드 → 위치 (실험 기록지의 load_position 열에 이 코드를 적는다)
        "C": {"name": "중앙", "x_mm": 600, "y_mm": 0},
        "E": {"name": "편심 (+y)", "x_mm": 600, "y_mm": 40},
    },
    "duration_s": 120,             # 실험 1회 계측 시간 [초]
}

# =============================================================================
# [A-3] Ground Truth 체계 (HW 합의 2026-10-12: D 코드의 뜻을 두 교량에서 통일)
# =============================================================================
# D0 = 정상, D1~D3 = 서로 다른 이상 유형 (위험도 단계 아님), L1~L3 = 같은 유형 안의 상대 심각도.
# ⚠️ damage_type, severity_level 은 정답이므로 평가에만 쓰고 모델 입력에는 절대 쓰지 않는다.
#    experiment_id 문자열(예: BOX_D1_L2_003)에도 정답이 들어 있으므로 특징으로 쓰지 않는다.
GROUND_TRUTH = {
    "normal_code": "D0",
    "normal_severity": "NORMAL",
    "damage_codes": ("D1", "D2", "D3"),
    "severity_levels": ("L1", "L2", "L3"),
    "damage_meaning": {"D1": "연결부 풀림", "D2": "지지조건 변경", "D3": "부재 강성 변경"},
}
LABEL_COLUMNS = ("damage_type", "severity_level")

# =============================================================================
# [A-4] 데이터 품질 플래그 (quality_flag)
# =============================================================================
#   OK   : 정상 측정
#   SAT  : 센서 측정 범위 초과 (포화)
#   DROP : 전송 누락 (value 비어 있음)
#   CAL  : 영점 조정 등 보정 중
QUALITY = {
    "flags": ("OK", "SAT", "DROP", "CAL"),
    "use": ("OK",),  # 기본 분석에 사용하는 플래그. 나머지는 제거하거나 별도 처리
}

# =============================================================================
# [A-5] 전처리 → Window → 특징 추출  ← 모두 개발용 초기값. 실제 데이터로 다시 조정
# =============================================================================
PREPROCESS = {
    # quality_flag 별 처리: "drop" = 값을 결측으로 바꿔 분석에서 제외, "keep" = 그대로 사용
    "quality_policy": {"SAT": "drop", "DROP": "drop", "CAL": "drop"},
    # 이 길이 이하의 짧은 결측 구간은 앞뒤 값으로 선형 보간. 더 긴 결측이 있는 Window 는 제외
    "max_gap_s": 0.2,
}

WINDOW = {
    # "tap"  : 타격 1번 = 구간 1개 (타격 직전 tap_pre_s 부터 length_s 동안). 감쇠비 계산에 필요 → 기본값
    # "fixed": 실험 시작부터 length_s 길이, overlap 겹침으로 일정하게 자름 (타격 여부와 무관)
    "mode": "tap",
    "length_s": 2.0,            # Window 길이 [초]
    "overlap": 0.5,             # (fixed 모드) 겹침 비율. 0.5 → 1초씩 이동
    "tap_pre_s": 0.05,          # (tap 모드) 타격 몇 초 전부터 구간을 시작할지
    "tap_min_gap_s": 1.0,       # (tap 모드) 타격과 타격 사이 최소 간격 [초]. 이보다 가까운 봉우리는 같은 타격
    "tap_threshold": 8.0,       # (tap 모드) 타격 판단 기준: 평소 흔들림(잡음)의 몇 배를 넘으면 타격으로 볼지
    "max_missing_ratio": 0.10,  # 원본 데이터가 이 비율보다 많이 빠진 Window 는 제외
}

# 노이즈 필터 (scipy Butterworth, 앞뒤 양방향 적용으로 시간 지연 없음).
# 처음에는 끄고(원본 사용) python src/preprocess.py --compare <experiment_id> 로 원본과 필터 결과를 비교한 뒤 결정
FILTER = {
    "enabled": False,
    "order": 4,
    # 센서 종류별 (필터 종류, 차단 주파수 Hz). 차단 주파수는 측정 속도의 절반(나이퀴스트)보다 작아야 함
    "acceleration": ("bandpass", (1.0, 150.0)),
    "load": ("lowpass", 5.0),
}

FEATURES = {
    "min_freq_hz": 2.0,          # 우세 진동수 탐색 하한 [Hz] (느린 흔들림·기준값 변화 무시)
    # 가속도 Band Energy 구간 [Hz]. 각 구간 에너지 / 전체 에너지 (0~1). 나이퀴스트를 넘는 부분은 자동으로 잘림
    "acc_bands_hz": [(2, 20), (20, 50), (50, 100), (100, 200)],
    # 감쇠비: 타격 직후 진동이 줄어드는 속도로 계산 (tap 모드에서만). 실측에서 안정적이지 않으면 False
    "use_damping": True,
    "damping_axis": "Z",         # 감쇠비를 계산할 축
    "damping_min_cycles": 3,     # 진폭이 20% 로 줄 때까지 최소 몇 번 떨려야 감쇠비를 계산할지 (너무 짧으면 부정확)
}

# =============================================================================
# [A-6] 데이터 분할 / 모델 입력 / 정상 기준 / Isolation Forest  ← 모두 개발용 초기값. 실제 데이터로 다시 조정
# =============================================================================
# 분할은 반드시 experiment_id 단위 (같은 실험의 Window 가 train 과 test 에 나뉘면 데이터 누수)
SPLIT = {
    # "column": experiments.csv 의 split 열 사용 (측정일(세션) 단위로 미리 정한 분할)
    # "random": D0 실험을 실험 단위로 무작위 분할 (아래 비율). D1~D3 는 전부 test
    "method": "column",
    "ratios": {"train": 0.4, "val": 0.2, "test": 0.4},
    "random_state": 42,
}

MODEL_INPUT = {
    # 하중·온도는 구조 이상이 아니라 실험 조건 → 모델 입력에서 빼고 '조건별 정상 기준'에 사용
    "exclude_sensor_types": ("load", "temperature"),
    # 빼는 특징 이름 (센서와 무관). acc mean 은 중력·영점 오차라 온도에 따라 흔들려 오탐 원인이 됨
    "exclude_feature_names": ("mean",),
    "exclude_features": (),          # 빼고 싶은 특징 열 이름을 직접 지정 (예: ("acc01_crest_x",))
}

BASELINE = {
    # 정상 기준을 나누는 조건: 하중 크기 × 하중 위치 (condition 열, 예: "1kg_E")
    # 온도 보정: train D0 에서 '온도 1°C 당 특징 변화량'을 구해 빼 준다 (공기 온도 기준, 선형 가정)
    "temperature_correction": True,
    # 온도 보정 안전장치: 정상 데이터가 적으면 '그날의 우연한 차이'를 온도 탓으로 잘못 계산하기 쉽다
    "temp_min_range_c": 2.0,     # train D0 의 온도 범위가 이보다 좁으면 보정하지 않음
    "temp_min_t": 3.0,           # 기울기가 추정 오차의 몇 배 이상으로 뚜렷한 특징에만 보정 적용
    # 정상 기준 표준편차: "pooled" = 조건을 합쳐 계산 (잡음·기준값 변화는 조건과 무관 → 안정적), "per_condition"
    "std": "pooled",
    "std_floor": 1e-6,           # 표준편차 하한 (거의 변하지 않는 특징으로 나눌 때 값이 폭주하는 것 방지)
}

IFOREST = {
    "n_estimators": 200,         # 트리 개수. 많을수록 점수가 안정적 (속도는 느려짐)
    "max_samples": "auto",       # 트리 1개가 보는 샘플 수 (auto = min(256, 전체))
    "contamination": "auto",     # 판정에 쓰지 않음. 정상 데이터만으로 학습하므로 '이상 비율' 가정이 맞지 않음
    "random_state": 42,
    # 판정 기준: val 정상 Window 점수의 이 분위수. 0.99 → 정상 데이터 오탐 약 1% 허용
    "threshold_quantile": 0.99,
    # 보조 규칙: 특징 하나라도 정상 기준에서 크게 벗어나면 이상. 기준 = val 정상 Window 의 '최대 |z|' 분위수.
    # (IF 는 학습 범위를 크게 벗어난 값도 점수가 일정 수준에서 멈추는 한계가 있어 이를 보완)
    "z_threshold_quantile": 0.99,
    # 0~1 Anomaly Score 변환 방식 (train.to_unit_score)
    #   "iforest_paper": Isolation Forest 원 논문 점수 2^(-평균경로/c). 정상 ≈ 0.5 이하, 클수록 이상
    #   "minmax_val"   : val 정상 점수의 1%~99% 범위를 0~1 로 늘린 값 (범위 밖은 0 또는 1)
    "score_method": "iforest_paper",
    "use_z_rule": True,          # 보조 규칙(특징 하나라도 크게 벗어나면 이상) 사용 여부
}

PREDICTION_LABELS = ("Normal", "Anomaly")

# 모니터링 증상 문구: 특징 이름 → 사람이 읽는 이름. 증상 = "<센서 종류> <축> <특징> 증가/감소 (±z σ)"
# 원인(D1~D3 유형) 판정이 아니라 '무엇이 평소와 다른가'를 알려주는 문구
SENSOR_TYPE_KO = {"acceleration": "가속도", "load": "하중", "temperature": "온도", "cross": "센서 간 비교"}
AXIS_KO = {"X": "길이방향(X)", "Y": "폭방향(Y)", "Z": "수직(Z)"}
FEATURE_KO = {
    "mean": "평균", "rms": "진동 크기(RMS)", "std": "변동폭(STD)", "max": "최댓값", "min": "최솟값",
    "p2p": "진폭(Peak-to-Peak)", "crest": "충격성(Crest)", "dom_freq": "진동수", "damping": "감쇠비",
    "band_2_20": "저주파 에너지(2~20Hz)", "band_20_50": "20~50Hz 에너지", "band_50_100": "50~100Hz 에너지",
    "band_100_200": "고주파 에너지(100~200Hz)",
    "amp_ratio_L4_L2_z": "L/4÷L/2 수직 진폭 비 (진동 형상)", "torsion_y_z_L2": "L/2 측면 폭방향÷수직 비 (비틀림)",
}

# 최종 결과 CSV 에 함께 넣을 특징 (AI② 위험도 모델 입력용). 센서 종류별 특징 이름.
# 가속도는 축마다 들어간다 (예: acc01_rms_z, acc02_dom_freq_y). 센서 간 비교 특징은 모두 들어간다.
OUTPUT_FEATURES = {
    "acceleration": ("rms", "p2p", "dom_freq", "damping"),
    "load": ("mean", "std"),
    "temperature": ("mean",),
}

# 연속 판정 규칙: 센서마다 최근 persist_n 개 Window 중 persist_k 개 이상이 이상이어야 최종 anomaly.
# 한두 번 튀는 오탐을 걸러낸다. 대신 탐지가 최소 persist_k 개 Window 만큼 늦어진다. (개발용 초기값)
DECISION = {
    "use_persistence": True,
    "persist_n": 3,   # tap 모드: 타격 3번 (약 15초), fixed 모드: 50% 겹침 2초 Window 3개 (약 4초)
    "persist_k": 2,
}

# AI② 전달 API: 구간(Window)마다 메시지 1건 (raw + AI① 결과) 을 HTTP POST (JSON) 로 보낸다.
# url 이 비어 있으면 보내지 않고 파일로만 저장한다 (AI② 주소가 정해지면 채운다).
AI2_API = {
    "url": os.environ.get("AI2_API_URL", ""),   # 예: "http://<AI② 서버>:8000/api/ai1/windows"
    "timeout_s": 5,
    # 재전송 (AI② 합의 2026-10-15): 네트워크 오류·5xx 면 최대 max_retries 번 다시 보냄 (간격 1·2·4초). 4xx 는 재전송 안 함.
    # AI② 는 experiment_id + window_id 로 중복 수신을 걸러 낸다.
    "max_retries": 3,
    "retry_backoff_s": 1.0,
    "schema_version": "1.0",
}

# AI① 수신 API (ai1_server.py): HW/BE 가 raw 를 POST /api/raw 로 보내면 판정 후 AI② 로 전송.
# 교량마다 서버 1개 (IOT_BRIDGE 로 교량 선택). 포트는 실행 시 --port 로 바꿀 수 있다.
AI1_SERVER = {
    "host": "0.0.0.0",
    "port": {"box": 8001, "truss": 8002},
    # 버퍼 끝에서 이 시간 [초] 이상 지난 구간만 판정 확정·전송 (뒤 데이터가 더 와도 구간이 바뀌지 않도록)
    "settle_s": 3.0,
}

# =============================================================================
# [A-7] 공개 데이터 변환 (convert_public.py)  ← AI①·AI② 합의서 v3 (2026-11-12) 기준
# =============================================================================
# 공개 교량 데이터를 RAW_COLUMNS 형식으로 바꿔 data/raw/public_raw/<dataset>/ 에 두고,
# 같은 신호를 HW 가속도계 수준으로 열화해 data/raw/public_deg/<dataset>/ 에 둔다.
ORIGINAL_DIR = ROOT / "data" / "original"   # 내려받은 원본 (수정하지 않음)
PUBLIC_SOURCES = ("public_raw", "public_deg")

# HW 가속도계 LIS2DW12 가정값 (고성능 + 저잡음 모드). 열화 순서: 잡음 → 양자화 → 범위 제한
LIS2DW12 = {
    "noise_ug_per_rthz": 90.0,   # 잡음 밀도 [µg/√Hz]
    "range_g": 2.0,              # 측정 범위 ±2g (넘으면 잘라내고 quality_flag = SAT)
    "bits": 14,                  # 분해능
    "seed": 20261109,            # 열화 난수 시드 (합의서 v3, 결과를 본 뒤 바꾸지 않음)
}

# HBTA (Hell Bridge Test Arena, 트러스교). Zenodo 10.5281/zenodo.14632942, CC BY 4.0
HBTA = {
    "file": "hbta/data_100Hz.h5",
    "sample_rate_hz": 100,
    # 진동기 노이즈 가진(P2, 수직) 측정만 사용. 스윕 가진은 주 진동수가 가진을 따라가므로 제외
    "records": ["MVS_P2_UDS_NM_Z_01", "MVS_P2_UDS_NM_Z_02"] + [f"MVS_P2_DS{i}_NM_Z_01" for i in range(1, 9)],
    "sensors": {"HBTA_ACC_01": "AG03", "HBTA_ACC_02": "AG05"},   # acc01 = AG03 (x=-7m, 약 0.30L), acc02 = AG05 (x=0, L/2)
    "axes": ["Y", "Z"],          # 벽면 가속도계에는 길이방향(X) 축이 없음
    "steady_s": (90.0, 930.0),   # 가진이 안정된 구간 [초]. 앞쪽 가진 상승 구간 제외
    "window_s": 10.24,           # 기본 Feature·AR(5) Window (겹침 없음)
    # 정상(UDS) 측정마다 Window 를 시간 순서대로 나눈 개수: AI① train / AI① val / AI② 보정 / test(블록 2개)
    "uds_split_windows": {"train": 33, "val": 16, "ai2_calib": 16, "test": (9, 8)},
    # 스윕 가진(SM) 측정 40개의 번호·ID 는 처리 전에 고정한다 (INFRIUM/AI-2#15, AI② 2026-11-19 요청)
    #   열화 난수 측정 번호: SM 측정 이름 알파벳 순서대로 11–50 (기존 NM 10개는 1–10 그대로)
    #   record_id HBTA_C_01–40, experiment_id HBTA_0019–0058: 시드 고정 순열 (정답이 드러나지 않게). 표는 results/public/hbta_sm_id_table.csv
    # 개발용 (AI1-public-1.2-dev): 정상 SM 6개. 사전 등록 전에 처리해도 되는 측정 (손상 정보 없음, P2 Z 정상 2개 제외)
    "dev": {
        "records": ["MVS_P1_UDS_SM_Y_01", "MVS_P1_UDS_SM_Y_02", "MVS_P1_UDS_SM_Z_01", "MVS_P1_UDS_SM_Z_02",
                    "MVS_P2_UDS_SM_Y_01", "MVS_P2_UDS_SM_Y_02"],
    },
    # 확인용 (AI1-public-1.2): AI② Truss v2 사전 등록서가 Push 된 뒤에만 채운다.
    # preregistration 이 비어 있으면 convert_public.py 가 확인용 측정을 변환하지 않는다 (사전 등록 전 결과 방지)
    "check": {
        "preregistration": "",        # 사전 등록서 위치 (예: INFRIUM/AI-2 커밋 주소)
        "records": [],                # 사전 등록서에 적힌 측정 이름
        "split": {},                  # 측정 이름 → ai1_split (사전 등록서대로, 없으면 "test")
        "steady_s": None,             # 사용할 구간 [초] (사전 등록서대로)
    },
}

# Ponte Moesa Campagnola (박스교). ETH 10.3929/ethz-b-000716850, CC BY-NC 4.0
# 시각은 원본 파일 그대로 (현지 시각, 시간대 표기 없음). 라벨은 공식 목록 Ambient.txt + 논문 표 1 (합의서 v3 3절)
MOESA = {
    "zips": {26: "moesa/Data Nov 26_27.zip", 27: "moesa/Data Nov 26_27.zip",
             28: "moesa/Data Nov 28_29.zip", 29: "moesa/Data Nov 28_29.zip"},
    "extract_dir": ROOT / "data" / "interim" / "moesa",   # .mat 압축 해제 위치 (원본 폴더는 건드리지 않음)
    "sample_rate_hz": 200,
    # MOESA_ACC_01 = 이동 센서 4번 (S5 에서 x=53.75m, 3/4L), MOESA_ACC_02 = 고정 센서 2번 (x=45.5m, L/2),
    # MOESA_ACC_03 = 고정 센서 3번 (x=78.5m, 세션 차이 확인용). 4번은 S5 측정에서만 사용
    "sensors": {"MOESA_ACC_01": 4, "MOESA_ACC_02": 2, "MOESA_ACC_03": 3},
    "window_s": 60.0, "overlap": 0.5,
    # 실험 블록: (측정 ID, 시작, 끝, 상태, split, 묶음, label_status). 시각은 "일 HH:MM:SS"
    # 묶음: A/B = 교차 검증 묶음(cv_group), 작업전·작업중·저녁·낮 = 보고용 구분(subset)
    "blocks": [
        ("AMB_18", "27 12:45:00", "27 14:40:00", "DS2", "train", "", "확실"),
        ("AMB_18", "27 14:40:00", "27 15:30:00", "DS2", "val", "", "확실"),
        ("AMB_18", "27 15:30:00", "27 16:20:00", "DS2", "ai2_calib", "", "확실"),
        ("AMB_18", "27 16:20:00", "27 17:00:00", "DS2", "test", "작업중", "확실"),
        ("AMB_16", "27 11:34:35", "27 11:41:00", "DS2", "test", "작업전", "확실"),
        ("AMB_17", "27 12:25:00", "27 12:40:00", "DS2", "test", "작업전", "확실"),
        ("AMB_28", "28 10:26:00", "28 11:04:00", "DS3", "test", "A", "확실"),
        ("AMB_27", "28 09:43:00", "28 09:53:00", "DS3", "test", "B", "확실"),
        ("AMB_29", "28 11:10:15", "28 11:25:15", "DS3", "test", "B", "확실"),
        ("AMB_30", "28 11:28:30", "28 11:35:10", "DS3", "test", "B", "확실"),
        ("AMB_35", "28 14:02:10", "28 14:07:20", "DS5", "test", "A", "확실"),
        ("AMB_36", "28 15:20:00", "28 15:22:00", "DS5", "test", "B", "확실"),     # DS6 절단 작업(14:13~15:20) 이후만
        ("AMB_36", "28 14:14:00", "28 15:22:00", "DS5", "test", "", "참고"),      # 표 S2 라벨대로 전체 (참고)
        ("AMB_37", "28 15:22:00", "28 15:29:45", "DS6", "test", "", "참고"),      # DS6 은 등급 판정에 쓰지 않음
        ("AMB_47", "29 08:56:00", "29 09:05:45", "DS7", "test", "A", "확실"),
        ("AMB_47", "29 09:05:45", "29 09:15:30", "DS7", "test", "B", "확실"),
        # 세션 차이 확인용 (고정 센서 2·3번만, AI① 모델 학습·판정에는 쓰지 않음)
        ("AMB_7", "26 16:50:08", "26 17:00:00", "DS2", "session", "저녁", "확실"),
        ("AMB_8", "26 17:21:00", "26 17:30:58", "DS2", "session", "저녁", "확실"),
        ("AMB_9", "26 17:50:00", "26 18:00:54", "DS2", "session", "저녁", "확실"),
        ("AMB_12", "27 09:13:00", "27 09:23:00", "DS2", "session", "낮", "확실"),
        ("AMB_13", "27 10:12:28", "27 10:17:00", "DS2", "session", "낮", "확실"),
        ("AMB_14", "27 10:35:50", "27 10:41:00", "DS2", "session", "낮", "확실"),
        ("AMB_15", "27 10:53:00", "27 10:55:40", "DS2", "session", "낮", "확실"),
    ],
    "setup": {"AMB_7": 3, "AMB_8": 4, "AMB_9": 6, "AMB_12": 1, "AMB_13": 2, "AMB_14": 3, "AMB_15": 4},   # 나머지는 S5
}

# Old ADA (강 Warren 트러스, 다른 교량 점검용). Mendeley 10.17632/sc8whx4pvm.2, CC BY 4.0
# 열 대응·단위는 Quqa & Landi (2023) + AI① 모드 형상 확인 (단위 m/s² 는 추정)
OLD_ADA = {
    "zip": "old_ada/Old_ADA_Bridge-damage_vibration_data.zip",
    "sample_rate_hz": 200,
    "sensors": {"OLDADA_ACC_01": 1, "OLDADA_ACC_02": 3},   # 진동 열 번호 (시간 열 다음부터 0): A1 = L/4, A3 = L/2
    "optical_exit_col": 11,                                # 차량 주행 데이터의 진출 광센서 열
    "free_decay_s": (1.0, 31.0),                           # 진출 후 자유 감쇠 Window [초]
    # 40km/h 정상 주행 분할 (주행 순서 번호): AI① train / AI② 기준 / test
    "int40_split": {"train": (1, 2, 3, 4), "ai2_calib": (5, 6, 7), "test": (8, 9, 10)},
}


if ACTIVE_BRIDGE not in BRIDGES:
    raise SystemExit(f"IOT_BRIDGE='{ACTIVE_BRIDGE}' 는 BRIDGES 에 없습니다. 가능한 값: {list(BRIDGES)}")


def validate(bridge_key=ACTIVE_BRIDGE):
    """설정값을 고친 뒤 흔한 실수(오타, 중복 ID, 범위 밖 값)를 실행 전에 잡아낸다."""
    bridge = BRIDGES[bridge_key]
    ids = [s["sensor_id"] for s in bridge["sensors"]]
    errors = []
    if len(ids) != len(set(ids)):
        errors.append(f"sensor_id 중복: {ids}")
    for s in bridge["sensors"]:
        if s["sensor_type"] not in SENSOR_TYPES:
            errors.append(f"{s['sensor_id']}: sensor_type '{s['sensor_type']}' 은(는) {SENSOR_TYPES} 중 하나여야 함")
        if not 0.0 <= s["x_ratio"] <= 1.0:
            errors.append(f"{s['sensor_id']}: x_ratio 는 0.0~1.0 사이여야 함")
        if s["sample_rate_hz"] <= 0:
            errors.append(f"{s['sensor_id']}: sample_rate_hz 는 0보다 커야 함")
        if not s.get("axes"):
            errors.append(f"{s['sensor_id']}: axes 가 비어 있음 (3축 = ['X','Y','Z'], 1축 = ['NA'])")
    for name, num, den in bridge.get("cross_features", []):
        for sid, axis in (num, den):
            if sid not in ids:
                errors.append(f"cross_features '{name}': 센서 '{sid}' 가 sensors 에 없음")
    types = [s["sensor_type"] for s in bridge["sensors"]]
    if "acceleration" not in types:
        errors.append("가속도 센서가 하나도 없음")
    for code in bridge.get("damage_types", {}):
        if code not in GROUND_TRUTH["damage_codes"]:
            errors.append(f"damage_types 의 '{code}' 는 {GROUND_TRUTH['damage_codes']} 중 하나여야 함")
    if errors:
        raise ValueError("config.py 설정 오류:\n  - " + "\n  - ".join(errors))
