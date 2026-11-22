"""
LUMO 분할 ZIP 에서 필요한 10분 블록만 부분 다운로드 (HTTP Range). AI② Truss v4 사전 등록서 3절.

LUMO 월 폴더(structural_data/<yyyy>/<mm>/)는 ZIP 하나를 10GB 씩 자른 <mm>.zip.001 … 이다 (압축 없음).
  index  : 각 달 마지막 분할 파일 끝의 ZIP 목록(central directory)만 받아 블록 파일 이름·위치·크기·CRC32 표를 만든다
           (값은 읽지 않음) → data/original/lumo/zip_index/<yyyymm>.csv
  fetch  : 표에서 위치를 찾아 블록 파일 하나를 받아 CRC32 확인 (분할 경계에 걸치면 두 파일에서 이어 받음)

실행: python src/lumo_fetch.py index
"""
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

import pandas as pd

from config import ORIGINAL_DIR

BASE = "https://data.uni-hannover.de/vault/isd/wernitz/lumo/structural_data"
INDEX_DIR = ORIGINAL_DIR / "lumo" / "zip_index"
MONTHS = ["202008", "202009", "202010", "202011", "202012", "202101", "202102", "202103", "202104", "202105", "202106", "202107"]
UA = {"User-Agent": "python-urllib (research data client)"}


def part_url(ym, k):
    return f"{BASE}/{ym[:4]}/{ym[4:]}/{ym[4:]}.zip.{k:03d}"


def head_size(url):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def get_range(url, start, end):
    """[start, end] 바이트 (양 끝 포함)"""
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        if r.status != 206:
            raise RuntimeError(f"Range 미지원 응답 {r.status}: {url}")
        return r.read()


def n_parts(ym):
    """sha1sum 목록에서 그 달의 분할 수"""
    y = ym[:4]
    lines = (ORIGINAL_DIR / "lumo" / "meta_struct" / f"sha1sum_{y}.txt").read_text().splitlines()
    return max(int(l.rsplit(".", 1)[1]) for l in lines if f"./{ym[4:]}/{ym[4:]}.zip." in l)


class Month:
    """한 달의 분할 ZIP 을 이어 붙인 하나의 바이트 열로 다룬다 (전역 위치 → 분할 파일·위치)"""

    def __init__(self, ym):
        self.ym, self.n = ym, n_parts(ym)
        self.part_size = head_size(part_url(ym, 1))
        self.last_size = head_size(part_url(ym, self.n))
        self.total = self.part_size * (self.n - 1) + self.last_size

    def read(self, start, length):
        out, pos, end = [], start, start + length
        while pos < end:
            k = pos // self.part_size + 1
            off = pos - (k - 1) * self.part_size
            size = self.part_size if k < self.n else self.last_size
            take = min(end - pos, size - off)
            out.append(get_range(part_url(self.ym, k), off, off + take - 1))
            pos += take
        return b"".join(out)

    def parts_of(self, start, length):
        return sorted({start // self.part_size + 1, (start + length - 1) // self.part_size + 1})


def central_directory(m):
    """ZIP 끝의 (Zip64) End of Central Directory → 목록 항목 표"""
    tail_len = min(1 << 20, m.total)
    tail = m.read(m.total - tail_len, tail_len)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("EOCD 없음")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    j = tail.rfind(b"PK\x06\x06")   # Zip64 EOCD
    if j >= 0:
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = m.read(cd_off, cd_size)
    rows, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (method, crc, csize, usize, nlen, xlen, clen, lho) = (
            struct.unpack("<H", cd[p + 10:p + 12])[0], *struct.unpack("<III", cd[p + 16:p + 28]),
            *struct.unpack("<HHH", cd[p + 28:p + 34]), struct.unpack("<I", cd[p + 42:p + 46])[0])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):   # Zip64 확장: 0xFFFFFFFF 인 값만 순서대로
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                vals, r = list(struct.unpack(f"<{hlen // 8}Q", extra[q + 4:q + 4 + (hlen // 8) * 8])), 0
                if usize == 0xFFFFFFFF:
                    usize, r = vals[r], r + 1
                if csize == 0xFFFFFFFF:
                    csize, r = vals[r], r + 1
                if lho == 0xFFFFFFFF:
                    lho, r = vals[r], r + 1
            q += 4 + hlen
        rows.append({"name": name, "method": method, "crc32": f"{crc:08x}", "compressed_size": csize,
                     "uncompressed_size": usize, "local_header_offset": lho})
        p += 46 + nlen + xlen + clen
    return pd.DataFrame(rows)


def build_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for ym in MONTHS:
        path = INDEX_DIR / f"{ym}.csv"
        if path.exists():
            continue
        m = Month(ym)
        cd = central_directory(m)
        cd["part_size"], cd["n_parts"], cd["total_size"] = m.part_size, m.n, m.total
        cd.to_csv(path, index=False)
        print(f"{ym}: 분할 {m.n}개, 항목 {len(cd)}개, 압축 방식 {sorted(cd['method'].unique())}")


def fetch_member(ym, row, dest):
    """목록 한 항목(블록 파일)을 받아 dest 에 저장, CRC32 확인. 반환: (CRC 일치, 받은 전역 바이트 범위, 분할 번호)"""
    m = Month.__new__(Month)
    m.ym, m.n, m.part_size = ym, int(row["n_parts"]), int(row["part_size"])
    m.last_size = int(row["total_size"]) - m.part_size * (m.n - 1)
    m.total = int(row["total_size"])
    lho = int(row["local_header_offset"])
    head = m.read(lho, 30)
    nlen, xlen = struct.unpack("<HH", head[26:30])
    start = lho + 30 + nlen + xlen
    data = m.read(start, int(row["compressed_size"]))
    if int(row["method"]) != 0:
        raise RuntimeError(f"압축된 항목은 지원하지 않음: {row['name']}")
    Path(dest).write_bytes(data)
    ok = f"{zlib.crc32(data) & 0xFFFFFFFF:08x}" == row["crc32"]
    return ok, (start, start + len(data) - 1), m.parts_of(start, len(data))


if __name__ == "__main__":
    if sys.argv[1:] == ["index"]:
        build_index()
