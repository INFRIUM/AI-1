"""
AI① 수신 API: HW/BE 가 raw 를 보내면 → 이상탐지 → 구간마다 메시지 (raw + AI① 결과) 를 AI② API 로 전송.

  HW(ESP32) / BE ──POST /api/raw──▶ AI① (이 서버) ──POST (config.AI2_API url)──▶ AI②

[받는 형식] POST /api/raw   (Content-Type: application/json)
  {
    "session_id": "BOX_20261015_01",   측정 세션 ID (같은 측정이면 같은 값으로 계속 보냄). 정답(D·L)을 넣지 말 것
    "bridge_type": "BOX",              (선택) 이 서버의 교량과 다르면 거부
    "load_position": "C",              (선택) 하중 위치 코드. 센서로 측정되지 않으므로 함께 알려 줌. 하중 0 kg 이면 생략
    "end": false,                      측정이 끝났으면 true → 남은 구간까지 모두 판정·전송하고 세션 종료

    raw 는 아래 두 형식 중 하나 (자세한 설명: results/<source>/for_HW/HW_API_SPEC.md)
    (A) 가벼운 형식 (권장, ESP32 용) — 센서별 배열
    "sensors": {
      "BOX_ACC_01": {"t0_us": 1792836000000000, "axes": {"X": [...], "Y": [...], "Z": [...]}},
      "LOAD_01":    {"t0_us": 1792836000000000, "axes": {"NA": [...]}}
    }
      - 샘플 시각 = t0_us + index × 1e6 / sample_rate_hz (간격이 고르지 않으면 "timestamp_us": [...] 로 직접)
      - 선택: "sample_rate_hz" (없으면 등록값), "quality": {축: {플래그: [index...]}} (없으면 전부 OK)
    (B) 행 형식 — 데이터 사양 RAW_COLUMNS, 1행 = 센서 1개 × 축 1개 × 시각 1개
    "rows": [{"sensor_id": "BOX_ACC_01", "timestamp_us": 1792836000000000, "axis": "Z", "value": 9.81,
              "unit": "m/s^2", "sample_rate_hz": 400, "quality_flag": "OK"}, ...]
      - unit / sample_rate_hz 를 빼면 센서 등록 정보로 채운다. quality_flag 를 빼면 "OK", axis 를 빼면 "NA".
  }
  - 몇 초 분량씩 나눠 보내면 된다. 서버가 세션별로 모아 두고, 새로 확정된 구간만 판정해 AI② 로 보낸다.
  - 같은 데이터를 다시 보내도 (센서·축·시각이 같으면) 한 번만 쓰인다 → 네트워크 오류 시 재전송해도 안전.

[처리]
  1. 세션 버퍼에 raw 추가 → 버퍼 전체를 다시 판정 (predict.judge_raw: 타격 구간 찾기 → 특징 → Isolation Forest → 연속 판정)
  2. 버퍼 끝에서 settle_s 초 이상 지난 구간 중 아직 안 보낸 구간만 확정 (뒤 데이터로 구간이 바뀌지 않도록)
  3. 확정 구간마다 메시지 1건 (ai2_message.build_messages) → results/<source>/<bridge>/live/ 에 기록
  4. 전송 전용 스레드가 순서대로 AI② API 로 POST. 네트워크 오류·5xx 면 최대 3회 재전송 (config.AI2_API)
     끝까지 실패하면 live/ai2_send_failed.jsonl 에 남긴다

[응답] {"session_id", "buffer_s", "new_windows", "queued_to_ai2", "windows": [{"window_id", "prediction", "confidence"}]}

GET /api/health → 서버 상태

실행: python src/ai1_server.py                       (box, 포트 8001)
      $env:IOT_BRIDGE="truss"; python src/ai1_server.py   (truss, 포트 8002)
      python src/ai1_server.py --port 9000              (포트 지정)
AI② 주소: config.AI2_API["url"] 또는 환경변수 AI2_API_URL. 비어 있으면 전송하지 않고 live/ 에 기록만 한다.
"""
import json
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from ai2_message import build_messages, load_confidence, post_one
from config import ACTIVE_BRIDGE, AI1_SERVER, AI2_API, BRIDGES, EXPERIMENT, RESULTS_DIR, WINDOW
from predict import judge_raw, load_bundle

BRIDGE = BRIDGES[ACTIVE_BRIDGE]
SENSORS = {s["sensor_id"]: s for s in BRIDGE["sensors"]}
LIVE_DIR = RESULTS_DIR / ACTIVE_BRIDGE / "live"
SAME_WINDOW_US = WINDOW["tap_min_gap_s"] / 2 * 1e6  # 이 안에서 시작하는 구간은 이미 보낸 구간과 같은 것으로 본다

BUNDLE = load_bundle()
CALIB = load_confidence(ACTIVE_BRIDGE)
LOCK = threading.Lock()
SESSIONS = {}  # session_id → {"chunks": [DataFrame], "sent": [window_start_us], "load_position": str}
OUTBOX = queue.Queue()  # AI② 로 보낼 메시지 (순서대로 1건씩 전송)
FAILED_PATH = LIVE_DIR / "ai2_send_failed.jsonl"
STATS = {"sent": 0, "failed": 0}


def sender():
    """AI② 전송 전용 스레드. 재전송(최대 max_retries 회)으로 오래 걸려도 raw 수신은 멈추지 않는다.
    끝까지 실패한 메시지는 FAILED_PATH 에 그대로 남긴다 (나중에 다시 보낼 수 있음)."""
    while True:
        m = OUTBOX.get()
        error = post_one(m)
        if error is None:
            STATS["sent"] += 1
        else:
            STATS["failed"] += 1
            print(f"[{m['experiment_id']}] {m['window_id']} AI② 전송 실패: {error}", flush=True)
            FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(FAILED_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"error": error, "message": m}, ensure_ascii=False) + "\n")
        OUTBOX.task_done()


class BadRequest(Exception):
    pass


def to_frame(rows):
    """받은 rows → RAW_COLUMNS 형식 DataFrame. 빠진 unit·sample_rate_hz 는 센서 등록 정보로 채움."""
    if not isinstance(rows, list) or not rows:
        raise BadRequest("rows 가 비어 있습니다.")
    df = pd.DataFrame(rows)
    missing = {"sensor_id", "timestamp_us", "value"} - set(df.columns)
    if missing:
        raise BadRequest(f"rows 에 필수 항목이 없습니다: {sorted(missing)}")
    unknown = set(df["sensor_id"]) - set(SENSORS)
    if unknown:
        raise BadRequest(f"등록되지 않은 센서: {sorted(unknown)} (이 서버 교량: {BRIDGE['bridge_type']})")
    for col, default in (("axis", "NA"), ("quality_flag", "OK")):
        df[col] = df[col].fillna(default) if col in df else default
    for col, key in (("unit", "unit"), ("sample_rate_hz", "sample_rate_hz")):
        fill = df["sensor_id"].map(lambda s: SENSORS[s][key])
        df[col] = df[col].fillna(fill) if col in df else fill
    df["axis"] = df["axis"].astype(str)
    df["timestamp_us"] = df["timestamp_us"].astype("int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float32")
    return df[["sensor_id", "timestamp_us", "axis", "value", "unit", "sample_rate_hz", "quality_flag"]]


def sensors_to_frame(sensors):
    """가벼운 형식 (센서별 배열) → rows 와 같은 DataFrame.
    {"<센서 ID>": {"t0_us", "axes": {"X": [...], ...}, (선택) "sample_rate_hz", "timestamp_us": [...], "quality": {축: {플래그: [index]}}}}
    샘플 시각 = timestamp_us 배열이 있으면 그 값, 없으면 t0_us + index × 1e6 / sample_rate_hz"""
    if not isinstance(sensors, dict) or not sensors:
        raise BadRequest("sensors 가 비어 있습니다.")
    unknown = set(sensors) - set(SENSORS)
    if unknown:
        raise BadRequest(f"등록되지 않은 센서: {sorted(unknown)} (이 서버 교량: {BRIDGE['bridge_type']})")
    frames = []
    for sid, entry in sensors.items():
        reg = SENSORS[sid]
        axes = entry.get("axes") if isinstance(entry, dict) else None
        if not isinstance(axes, dict) or not axes:
            raise BadRequest(f"{sid}: axes 가 없습니다.")
        bad_axes = set(axes) - set(reg["axes"])
        if bad_axes:
            raise BadRequest(f"{sid}: 축 {sorted(bad_axes)} 은 없습니다 (가능: {reg['axes']}).")
        lengths = {len(v) for v in axes.values()}
        if len(lengths) != 1:
            raise BadRequest(f"{sid}: 축마다 값 개수가 다릅니다 {sorted(lengths)}.")
        n = lengths.pop()
        rate = float(entry.get("sample_rate_hz") or reg["sample_rate_hz"])
        if entry.get("timestamp_us") is not None:
            ts = pd.array(entry["timestamp_us"], dtype="int64")
            if len(ts) != n:
                raise BadRequest(f"{sid}: timestamp_us 개수({len(ts)})가 값 개수({n})와 다릅니다.")
        elif "t0_us" in entry:
            ts = int(entry["t0_us"]) + (pd.RangeIndex(n) * (1e6 / rate)).round().astype("int64")
        else:
            raise BadRequest(f"{sid}: t0_us 또는 timestamp_us 가 필요합니다.")
        quality = entry.get("quality") or {}
        for axis, values in axes.items():
            flags = pd.Series("OK", index=range(n))
            for flag, idx in (quality.get(axis) or {}).items():
                flags.iloc[[i for i in idx if 0 <= i < n]] = flag
            frames.append(pd.DataFrame({
                "sensor_id": sid, "timestamp_us": list(ts), "axis": str(axis),
                "value": pd.to_numeric(pd.Series(values, dtype=object), errors="coerce").astype("float32"),
                "unit": reg["unit"], "sample_rate_hz": rate, "quality_flag": flags.to_numpy(),
            }))
    return pd.concat(frames, ignore_index=True)


def handle_raw(body):
    session_id = str(body.get("session_id") or "").strip()
    if not session_id:
        raise BadRequest("session_id 가 필요합니다.")
    if body.get("bridge_type") and body["bridge_type"] != BRIDGE["bridge_type"]:
        raise BadRequest(f"이 서버는 {BRIDGE['bridge_type']} 교량용입니다 (받은 값: {body['bridge_type']}).")
    position = body.get("load_position") or ""
    if position and position not in EXPERIMENT["load_positions"]:
        raise BadRequest(f"load_position '{position}' 는 {list(EXPERIMENT['load_positions'])} 중 하나여야 합니다.")
    end = bool(body.get("end", False))
    if body.get("sensors"):  # 가벼운 형식 (센서별 배열, 권장)
        chunk = sensors_to_frame(body["sensors"])
    elif body.get("rows") or not end:  # 행 형식
        chunk = to_frame(body.get("rows"))
    else:  # 데이터 없이 end 만
        chunk = None

    with LOCK:
        s = SESSIONS.setdefault(session_id, {"chunks": [], "sent": [], "load_position": position})
        if chunk is not None:
            s["chunks"].append(chunk)
        if position:
            s["load_position"] = position
        if not s["chunks"]:
            SESSIONS.pop(session_id, None)
            raise BadRequest("받은 raw 가 없습니다.")
        raw = pd.concat(s["chunks"], ignore_index=True).drop_duplicates(["sensor_id", "axis", "timestamp_us"], keep="last")
        s["chunks"] = [raw]
        buffer_end = int(raw["timestamp_us"].max())
        buffer_s = (buffer_end - int(raw["timestamp_us"].min())) / 1e6

        final, _ = judge_raw(raw, session_id, s["load_position"], BUNDLE)
        messages = []
        if final is not None:
            settled = final if end else final[final["window_end_us"] <= buffer_end - AI1_SERVER["settle_s"] * 1e6]
            new = {int(t) for t in settled["window_start_us"]
                   if all(abs(int(t) - sent) > SAME_WINDOW_US for sent in s["sent"])}
            if new:
                messages = build_messages(final, raw, ACTIVE_BRIDGE, CALIB, only=new, to_raw_end=end)
                s["sent"].extend(sorted(new))
        if end:
            SESSIONS.pop(session_id, None)

    if messages:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(LIVE_DIR / f"{session_id}_ai2_messages.jsonl", "a", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    for m in messages:
        print(f"[{session_id}] {m['window_id']} {m['ai1']['prediction']:<7} confidence={m['ai1']['confidence']}", flush=True)
        if AI2_API["url"]:
            OUTBOX.put(m)
    return {
        "session_id": session_id,
        "buffer_s": round(buffer_s, 2),
        "session_ended": end,
        "new_windows": len(messages),
        "queued_to_ai2": len(messages) if AI2_API["url"] else 0,
        "ai2_url_set": bool(AI2_API["url"]),
        "windows": [{"window_id": m["window_id"], "window_start_us": m["window_start_us"],
                     "prediction": m["ai1"]["prediction"], "confidence": m["ai1"]["confidence"]} for m in messages],
    }


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") == "/api/health":
            with LOCK:
                sessions = {k: len(v["sent"]) for k, v in SESSIONS.items()}
            self.reply(200, {"status": "ok", "bridge_type": BRIDGE["bridge_type"], "ai2_url": AI2_API["url"] or None,
                             "schema_version": AI2_API["schema_version"], "open_sessions": sessions,
                             "ai2_send": {**STATS, "waiting": OUTBOX.qsize()}})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/api/raw":
            return self.reply(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self.reply(200, handle_raw(body))
        except (BadRequest, json.JSONDecodeError, ValueError, TypeError) as e:
            self.reply(400, {"error": str(e)})
        except Exception as e:  # 판정 중 오류: 서버는 계속 동작
            self.reply(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        pass


def main():
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else AI1_SERVER["port"][ACTIVE_BRIDGE]
    server = ThreadingHTTPServer((AI1_SERVER["host"], port), Handler)
    print(f"AI① 수신 API ({BRIDGE['bridge_type']}) → http://{AI1_SERVER['host']}:{port}/api/raw")
    print(f"AI② 전송 주소: {AI2_API['url'] or '없음 (live/ 폴더에 기록만)'}")
    if CALIB is None:
        print("⚠️ confidence.json 이 없어 confidence 는 null 로 나갑니다. export_for_ai2.py 를 먼저 실행하세요.")
    threading.Thread(target=sender, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
