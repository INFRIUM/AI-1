# 공개 데이터 원본 파일 목록

받은 파일을 아래 폴더에 그대로 넣는다. ZIP 은 풀지 않고, 파일 이름도 바꾸지 않는다.
이 폴더는 원본 보관용이다. 코드가 읽는 변환 결과는 `data/raw/real/` 에 따로 둔다.

합계 약 4.5 GB

## moesa/ : Ponte Moesa Campagnola (박스교, 학습)

출처: https://doi.org/10.3929/ethz-b-000716850

| 파일 (사이트 표시 이름) | 크기 | 용도 |
|---|---|---|
| General Information of the Dataset (Text) | 1.82 KB | 기본 설명 |
| General organization of the Dataset (PDF) | 765.75 KB | 폴더 구조, 측정 배치 확인 |
| Acceleration data, first and second day (ZIP) | 1011.37 MB | 측정 데이터 |
| Acceleration data, third and fourth day (ZIP) | 920.67 MB | 측정 데이터 |
| Matlab scripts and complementary information (ZIP) | 3.19 KB | 데이터 읽는 방법, 보충 정보 |
| LICENSE.txt | 1.13 KB | 라이선스 기록 |

받지 않는 파일:
- Distributed optical fibre sensor data (100.92 MB): HW 에 변형률 센서가 없어 쓰지 않는다.
- Plans, constructions details and pictures (313.55 MB): 보류. PDF 로 L/4·L/2 위치를 정할 수 없을 때만 받는다.

## hbta/ : Hell Bridge Test Arena (트러스교, 학습)

출처: https://zenodo.org/records/14632942 (DOI 10.5281/zenodo.14632942)

| 파일 | 크기 | md5 | 용도 |
|---|---|---|---|
| data_100Hz.h5 | 2.5 GB | 43406591cd7f8b8bd1b0b868d3312e38 | 측정 데이터 전체 |
| Drawings.pdf | 17.9 MB | 63d8de6902f4fc8f0e4abe22b02f9717 | 경간 내 실제 위치 확인 |
| Sensor layout.pdf | 147.2 kB | 251668e09ff33f03c9ff19aa93c0b020 | L/4·L/2 센서 번호 확인 |
| main.py | 1.4 kB | b4bb7f5676eec6d64f91f27f89e8c0d9 | h5 읽기 예제 |
| functions.py | 6.6 kB | 847e39a27fffd03e100b609f54e1f1c9 | h5 읽기 예제 |

## old_ada/ : Old ADA Bridge (트러스교, 다른 교량 검증용)

출처: https://data.mendeley.com/datasets/sc8whx4pvm/2

| 파일 | 크기 | 용도 |
|---|---|---|
| .mat 파일 1개 | 58.8 MB | 측정 데이터 |

사이트 파일 목록을 직접 확인하지 못했다 (크기는 AI② 문서 기준). 설명 문서가 더 있으면 같이 넣는다.

## 주의

- 데이터셋에 든 코드(`main.py`, `functions.py`, Matlab 스크립트)는 내용을 확인한 뒤 실행한다.
- Ponte Moesa 는 CC BY-NC 4.0 이라 비상업적 이용만 가능하다. HBTA 와 Old ADA 는 CC BY 4.0 이다 (AI② 문서 기준).
