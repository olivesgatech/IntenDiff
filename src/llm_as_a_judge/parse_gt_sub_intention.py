import os

TEST_SPLIT = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/darai/splits/test_split.txt"
GT_DIR     = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/darai/groundTruth_img"
OUT_DIR    = "/home/hice1/skim3513/scratch/causdiff/datasets/darai/gt_sub_intention"

STRIDE = 15  # 15행마다 1개씩

def extract_action_name(filename_with_ext: str) -> str:
    """
    예: '08_4_camera_1_fps_15_Cleaning dishes.txt'
    -> 마지막 '_' 이후 'Cleaning dishes' -> 공백을 '_'로 -> 'Cleaning_dishes'
    """
    base = os.path.basename(filename_with_ext).strip()
    if base.lower().endswith(".txt"):
        base = base[:-4]
    action_raw = base.rsplit("_", 1)[-1] if "_" in base else base
    return action_raw.replace(" ", "_")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(TEST_SPLIT, "r", encoding="utf-8") as f:
        vid_lines = [ln.strip() for ln in f if ln.strip()]

    total_videos = 0
    total_written = 0
    total_missing_gt = 0

    for vid_line in vid_lines:
        total_videos += 1  # 이 인덱스를 파일명에 사용

        base_no_ext = vid_line[:-4] if vid_line.lower().endswith(".txt") else vid_line
        gt_name = f"{base_no_ext}_1.txt"
        gt_path = os.path.join(GT_DIR, gt_name)

        if not os.path.exists(gt_path):
            total_missing_gt += 1
            print(f"[WARN] missing groundTruth: {gt_path}")
            continue

        action = extract_action_name(vid_line)
        out_path = os.path.join(OUT_DIR, f"{action}_{total_videos}_sub_intention.txt")

        written_this_video = 0
        with open(gt_path, "r", encoding="utf-8") as gtf, open(out_path, "w", encoding="utf-8") as out_fh:
            for i, row in enumerate(gtf):
                if i % STRIDE != 0:
                    continue

                row = row.strip()
                if not row:
                    continue

                parts = row.split(",")
                if len(parts) < 2:
                    continue

                val = parts[1].strip()
                out_fh.write(val + "\n")
                total_written += 1
                written_this_video += 1

        # 참고 로그
        # print(f"[OK] {out_path}  ({written_this_video} lines)")

    print("Done.")
    print(f"videos processed: {total_videos}")
    print(f"missing groundTruth files: {total_missing_gt}")
    print(f"total lines written: {total_written}")

if __name__ == "__main__":
    main()
