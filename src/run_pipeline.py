"""
전체 파이프라인을 교량별로 순서대로 실행.

순서: preprocess → train → predict → evaluate  (교량마다)
      → 마지막에 export_for_ai2 (AI② 전달용 폴더 results/<source>/for_AI2/ 갱신)

실행: python src/run_pipeline.py            ← config.py 의 BRIDGES 에 있는 모든 교량
      python src/run_pipeline.py truss      ← 지정한 교량만
"""
import os
import subprocess
import sys
from pathlib import Path

from config import BRIDGES

SRC = Path(__file__).resolve().parent
STEPS = ["preprocess.py", "train.py", "predict.py", "evaluate.py"]


def main():
    bridges = sys.argv[1:] or list(BRIDGES)
    for bridge in bridges:
        env = {**os.environ, "IOT_BRIDGE": bridge}
        for step in STEPS:
            print(f"\n===== [{bridge}] {step} =====", flush=True)
            result = subprocess.run([sys.executable, str(SRC / step)], env=env)
            if result.returncode != 0:
                raise SystemExit(f"[{bridge}] {step} 실패 (종료 코드 {result.returncode})")

    print("\n===== export_for_ai2.py =====", flush=True)
    if subprocess.run([sys.executable, str(SRC / "export_for_ai2.py")]).returncode != 0:
        raise SystemExit("export_for_ai2.py 실패")
    print(f"\n완료: {', '.join(bridges)}")


if __name__ == "__main__":
    main()
