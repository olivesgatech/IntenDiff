import os
import re
import json
from glob import glob
from typing import Optional, Tuple, Dict, Iterable

DIRS = [
    "/home/hice1/skim3513/scratch/causdiff/outputs/llm_as_a_judge/idefics-result",
    "/home/hice1/skim3513/scratch/causdiff/outputs/llm_as_a_judge/judge_results_local_vlm",
    "/home/hice1/skim3513/scratch/causdiff/outputs/llm_as_a_judge/llava-result",
]

WIN_RE  = re.compile(r"\bwinner\s*:\s*['\"]?(A|B|TIE)['\"]?", re.IGNORECASE)
CONF_RE = re.compile(r"\bconfidence\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

def parse_from_raw_text(raw_text: str) -> Tuple[Optional[str], Optional[float]]:
    if not raw_text:
        return None, None
    m1 = WIN_RE.search(raw_text)
    m2 = CONF_RE.search(raw_text)
    winner = m1.group(1).upper() if m1 else None
    conf = float(m2.group(1)) if m2 else None
    return winner, conf

def normalize_winner(w: Optional[str]) -> Optional[str]:
    if w is None:
        return None
    w = str(w).upper().strip()
    return w if w in ("A", "B", "TIE") else None

def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue

def extract_winner_conf_strict(rec: dict):
    """
    정책:
    - 정상 필드(A/B/TIE + confidence)면 사용
    - 단, parse_failed sentinel(=non-JSON output or TIE/0)인 경우:
        raw_text에서 winner/confidence를 복구할 수 있을 때만 사용
        복구 못 하면 skip (분모에서 제외)
    """
    def normalize_winner(w):
        if w is None:
            return None
        w = str(w).upper().strip()
        return w if w in ("A", "B", "TIE") else None

    w = normalize_winner(rec.get("winner", None))
    c = rec.get("confidence", None)
    try:
        c = float(c) if c is not None else None
    except Exception:
        c = None

    rationale = str(rec.get("rationale", "")).strip().lower()
    raw_text = rec.get("raw_text", "") or ""

    # 실패 sentinel 감지
    parse_failed = False
    if "non-json" in rationale:
        parse_failed = True
    if w == "TIE" and (c == 0.0 or c is None):
        parse_failed = True

    # 1) 실패 sentinel이면 raw_text에서만 인정
    if parse_failed:
        w2, c2 = parse_from_raw_text(raw_text)
        w2 = normalize_winner(w2)
        if w2 is not None and c2 is not None:
            return w2, float(c2), True
        # 복구 못 하면 스킵
        return None, None, False

    # 2) 실패 sentinel이 아니면 필드 값 인정
    if w in ("A", "B", "TIE") and c is not None:
        return w, c, False

    # 3) 필드가 비어있으면 raw_text 복구 시도
    w2, c2 = parse_from_raw_text(raw_text)
    w2 = normalize_winner(w2)
    if w2 is not None and c2 is not None:
        return w2, float(c2), True

    return None, None, False


def aggregate_dir(d: str, skip_undefined_gt: bool = False) -> Dict[str, float]:
    files = sorted(glob(os.path.join(d, "**", "*.jsonl"), recursive=True))

    used = 0
    skipped = 0
    recovered = 0

    a_wins = 0
    conf_sum = 0.0
    conf_sum_a = 0.0
    conf_cnt_a = 0

    for fp in files:
        for rec in read_jsonl(fp):
            if skip_undefined_gt:
                if str(rec.get("candidate_B_gt", "")).strip().upper() == "UNDEFINED":
                    continue

            w, c, recov = extract_winner_conf_strict(rec)
            if w is None or c is None:
                skipped += 1
                continue

            used += 1
            if recov:
                recovered += 1

            conf_sum += c
            if w == "A":
                a_wins += 1
                conf_sum_a += c
                conf_cnt_a += 1

    a_win_rate = (a_wins / used) if used > 0 else 0.0
    conf_avg_all = (conf_sum / used) if used > 0 else 0.0
    conf_avg_a = (conf_sum_a / conf_cnt_a) if conf_cnt_a > 0 else 0.0

    return {
        "num_files": float(len(files)),
        "used_records": float(used),
        "skipped_records": float(skipped),
        "recovered_from_raw": float(recovered),
        "a_win_rate": a_win_rate,
        "confidence_avg_all": conf_avg_all,
        "confidence_avg_when_A": conf_avg_a,
    }

def main(skip_undefined_gt: bool = False):
    print("=== Per-folder summary ===")
    overall_used = 0
    overall_skipped = 0
    overall_recovered = 0
    overall_a_wins = 0
    overall_conf_sum = 0.0
    overall_conf_sum_a = 0.0
    overall_conf_cnt_a = 0

    for d in DIRS:
        stats = aggregate_dir(d, skip_undefined_gt=skip_undefined_gt)
        print(f"\n[{d}]")
        print(f"  jsonl files: {int(stats['num_files'])}")
        print(f"  used records: {int(stats['used_records'])}")
        print(f"  skipped (unparsable) records: {int(stats['skipped_records'])}")
        print(f"  recovered from raw_text: {int(stats['recovered_from_raw'])}")
        print(f"  A win rate: {stats['a_win_rate']:.4f}")
        print(f"  confidence avg (all used): {stats['confidence_avg_all']:.4f}")
        print(f"  confidence avg (when A): {stats['confidence_avg_when_A']:.4f}")

        # overall 합산은 정확히 하려고 다시 한 번 누적(동일 로직)
        files = sorted(glob(os.path.join(d, "**", "*.jsonl"), recursive=True))
        for fp in files:
            for rec in read_jsonl(fp):
                if skip_undefined_gt:
                    if str(rec.get("candidate_B_gt", "")).strip().upper() == "UNDEFINED":
                        continue
                w, c, recov = extract_winner_conf_strict(rec)
                if w is None or c is None:
                    overall_skipped += 1
                    continue
                overall_used += 1
                if recov:
                    overall_recovered += 1
                overall_conf_sum += c
                if w == "A":
                    overall_a_wins += 1
                    overall_conf_sum_a += c
                    overall_conf_cnt_a += 1

    a_win_rate = (overall_a_wins / overall_used) if overall_used > 0 else 0.0
    conf_avg_all = (overall_conf_sum / overall_used) if overall_used > 0 else 0.0
    conf_avg_a = (overall_conf_sum_a / overall_conf_cnt_a) if overall_conf_cnt_a > 0 else 0.0

    print("\n=== Overall summary (all 3 folders) ===")
    print(f"  used records: {overall_used}")
    print(f"  skipped (unparsable) records: {overall_skipped}")
    print(f"  recovered from raw_text: {overall_recovered}")
    print(f"  A win rate: {a_win_rate:.4f}")
    print(f"  confidence avg (all used): {conf_avg_all:.4f}")
    print(f"  confidence avg (when A): {conf_avg_a:.4f}")

if __name__ == "__main__":
    # GT가 UNDEFINED인 샘플을 통계에서 제외하고 싶으면 True로
    main(skip_undefined_gt=False)
