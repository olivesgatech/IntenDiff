from transformers import pipeline
import os
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch
from PIL import Image

CACHE_ROOT = "/home/hice1/skim3513/scratch/causdiff/hf_cache"

os.environ["HF_HOME"] = CACHE_ROOT
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_ROOT, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_ROOT, "datasets")
os.environ["XDG_CACHE_HOME"] = os.path.join(CACHE_ROOT, "xdg")
os.environ["HF_MODULES_CACHE"] = os.path.join(CACHE_ROOT, "modules")

# -----------------------------
# Paths
# -----------------------------
PHRASES_DIR = "/home/hice1/skim3513/scratch/causdiff/outputs/darai_l3/phrases_30"
GT_DIR      = "/home/hice1/skim3513/scratch/causdiff/datasets/darai/gt_sub_intention"
TEST_SPLIT  = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/darai/splits/test_split.txt"

RGB_ROOT    = "/home/hice1/skim3513/AIFirst_F24_data/darai/RGB_sd"
OUT_DIR     = "/home/hice1/skim3513/scratch/causdiff/outputs/llm_as_a_judge/judge_results_local_vlm"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Constants
# -----------------------------
STRIDE = 15
PHRASES_HEADER_SKIP = 2      # phrases 파일 상단 2줄 버림
FRAME_OFFSET_LINES = 2       # 본문 첫 줄(j=0)이 00030이 되도록 +2 (네 예시 그대로)

PHRASES_RE = re.compile(r"^phrases_(?P<action>.+?)_(?P<ts>\d{8}_\d{6})\.txt$")

import os
import re
import json
import torch
from PIL import Image
from dataclasses import dataclass

# torchvision 필요
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

@dataclass
class JudgeResult:
    winner: str
    confidence: float
    rationale: str
    raw_text: str

def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width  = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images

def pil_to_pixel_values(image: Image.Image, input_size=448, max_num=12, device="cuda", dtype=torch.float16):
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles], dim=0)  # (N,3,448,448)
    return pixel_values.to(dtype=dtype, device=device)

class InternVL25Judge:
    """
    InternVL2.5-2B는 transformers pipeline이 아니라
    AutoModel(trust_remote_code=True) + model.chat()로 inference 해야 함. :contentReference[oaicite:1]{index=1}
    """
    def __init__(
        self,
        model_id: str,
        cache_root: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
        max_num_tiles: int = 12,
        use_flash_attn: bool = False,
    ):
        from transformers import AutoTokenizer, AutoModel

        os.makedirs(cache_root, exist_ok=True)

        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_num_tiles = max_num_tiles

        # bf16 가능하면 bf16, 아니면 fp16
        if device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16 if device.startswith("cuda") else torch.float32

        # trust_remote_code=True 로 prompt 제거 + InternVL custom config 인식 :contentReference[oaicite:2]{index=2}
        self.model = AutoModel.from_pretrained(
            model_id,
            cache_dir=cache_root,
            torch_dtype=self.dtype if self.dtype != torch.float32 else None,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
        ).eval()

        if device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_root,
            trust_remote_code=True,
            use_fast=False,
        )

    def judge_pair(self, image_path: str, action: str, t_idx: int, cand_a: str, cand_b: str) -> JudgeResult:
        if not os.path.exists(image_path):
            return JudgeResult("TIE", 0.0, "image file missing", "")

        image = Image.open(image_path).convert("RGB")
        pixel_values = pil_to_pixel_values(
            image,
            input_size=448,
            max_num=self.max_num_tiles,
            device="cuda" if self.device.startswith("cuda") and torch.cuda.is_available() else "cpu",
            dtype=self.dtype if self.dtype != torch.float32 else torch.float32,
        )

        # InternVL 단일 이미지 질의는 question에 <image> 토큰이 필요 :contentReference[oaicite:3]{index=3}
        question = (
            "<image>\n"
            "You are a strict judge for intention (goal) descriptions conditioned on the image.\n"
            "Compare candidate A vs candidate B for the SAME moment.\n"
            "Rules:\n"
            "1) Use only what is visually supported by the image. Do not invent details.\n"
            "2) Prefer intention/goal statements over merely naming the action.\n"
            "3) If both are equally good/bad or insufficient, output TIE.\n"
            "Return JSON ONLY with keys: winner, confidence, rationale.\n"
            "winner must be one of A, B, TIE. confidence must be in [0,1].\n\n"
            f"Action category: {action}\n"
            f"Time index: {t_idx}\n"
            f"Candidate A: {cand_a}\n"
            f"Candidate B: {cand_b}\n"
        )

        generation_config = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)

        out = str(response).strip()

        m = re.findall(r"\{.*\}", out, flags=re.DOTALL)
        blob = m[-1] if m else out

        try:
            data = json.loads(blob)
            winner = str(data.get("winner", "TIE")).upper()
            if winner not in ("A", "B", "TIE"):
                winner = "TIE"
            conf = float(data.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
            rationale = str(data.get("rationale", "")).strip()
            return JudgeResult(winner, conf, rationale, out)
        except Exception:
            return JudgeResult("TIE", 0.0, "non-JSON output", out)


# -----------------------------
# Utils: file listing / parsing
# -----------------------------
def sorted_txt_files(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    files.sort()
    return files

def parse_action_from_phrases_filename(fname: str) -> str:
    m = PHRASES_RE.match(fname)
    if not m:
        raise ValueError(f"Unexpected phrases filename format: {fname}")
    return m.group("action")  # e.g., Cleaning_dishes

def action_us_to_space(action_us: str) -> str:
    return action_us.replace("_", " ")  # Cleaning dishes

def group_phrases_by_action(phrases_files: List[str]) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = {}
    for f in phrases_files:
        a = parse_action_from_phrases_filename(f)
        d.setdefault(a, []).append(f)
    for a in d:
        d[a].sort()
    return d

def get_gt_files_for_action(action_us: str) -> List[str]:
    files = [f for f in os.listdir(GT_DIR) if f.endswith(".txt") and f.startswith(action_us)]
    files.sort()
    return files

def strip_time_prefix(line: str) -> str:
    line = line.strip()
    return re.sub(r"^time:\s*\d+\s*", "", line).strip()

def read_phrases_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        all_lines = [ln.rstrip("\n") for ln in f]
    body = all_lines[PHRASES_HEADER_SKIP:] if len(all_lines) > PHRASES_HEADER_SKIP else []
    out = []
    for ln in body:
        ln = ln.strip()
        if not ln:
            continue
        out.append(strip_time_prefix(ln))
    return out

def read_gt_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_test_split_lines() -> List[str]:
    with open(TEST_SPLIT, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def pick_test_split_line_for_action(action_space: str, idx0: int, test_lines: List[str]) -> Optional[str]:
    # 네가 말한 대로: "Cleaning dishes"가 포함된 행만 찾아서 그중 idx0번째(0-based)
    matches = [ln for ln in test_lines if action_space in ln]
    if idx0 < 0 or idx0 >= len(matches):
        return None
    return matches[idx0]

def parse_camera_and_prefix(test_line: str) -> Tuple[int, int, str]:
    """
    예: '15_4_camera_2_fps_15_Cleaning dishes.txt'
    -> cam=2, fps=15, img_prefix='15_4_'
    """
    base = os.path.basename(test_line)
    if base.lower().endswith(".txt"):
        base = base[:-4]
    parts = base.split("_")
    if len(parts) < 6 or parts[2] != "camera" or parts[4] != "fps":
        raise ValueError(f"Unexpected test_split format: {test_line}")
    cam = int(parts[3])
    fps = int(parts[5])
    img_prefix = f"{parts[0]}_{parts[1]}_"
    return cam, fps, img_prefix

def image_path_for_step(action_space: str, cam: int, fps: int, img_prefix: str, j: int) -> str:
    """
    j: phrases 본문 라인 인덱스(0-based)
    frame = (j + FRAME_OFFSET_LINES) * STRIDE
    """
    frame = (j + FRAME_OFFSET_LINES) * STRIDE
    folder = os.path.join(RGB_ROOT, action_space, f"camera_{cam}_fps_{fps}")
    fname = f"{img_prefix}{frame:05d}.jpg"
    return os.path.join(folder, fname)

# -----------------------------
# Main runner
# -----------------------------
def run_judge(model_id: str, limit_phrases_files: Optional[int] = None):
    phrases_files = sorted_txt_files(PHRASES_DIR)
    if limit_phrases_files is not None:
        phrases_files = phrases_files[:limit_phrases_files]

    phrases_by_action = group_phrases_by_action(phrases_files)
    test_lines = load_test_split_lines()

    judge = InternVL25Judge(
        model_id="OpenGVLab/InternVL2_5-2B",
        cache_root=CACHE_ROOT,
        device="cuda",
        max_new_tokens=256,
        max_num_tiles=12,      # VRAM 부족하면 6으로 낮춰도 됨
        use_flash_attn=False,  # flash-attn 없으면 False가 안전
    )

    total_pairs = 0
    missing_gt = 0
    missing_test = 0

    for phrases_fname in phrases_files:
        action_us = parse_action_from_phrases_filename(phrases_fname)   # Cleaning_dishes
        action_space = action_us_to_space(action_us)                    # Cleaning dishes

        # action 내 phrases index (0-based)
        action_list = phrases_by_action[action_us]
        idx0 = action_list.index(phrases_fname)

        # GT match
        gt_files = get_gt_files_for_action(action_us)
        if idx0 >= len(gt_files):
            missing_gt += 1
            print(f"[WARN] GT match missing: action={action_us}, idx0={idx0}, gt_count={len(gt_files)}")
            continue
        gt_fname = gt_files[idx0]

        # test_split match
        test_line = pick_test_split_line_for_action(action_space, idx0, test_lines)
        if test_line is None:
            missing_test += 1
            print(f"[WARN] test_split match missing: action_space='{action_space}', idx0={idx0}")
            continue

        cam, fps, img_prefix = parse_camera_and_prefix(test_line)

        phrases_path = os.path.join(PHRASES_DIR, phrases_fname)
        gt_path      = os.path.join(GT_DIR, gt_fname)

        A_lines = read_phrases_lines(phrases_path)
        B_lines = read_gt_lines(gt_path)

        n = min(len(A_lines), len(B_lines))
        if n == 0:
            print(f"[WARN] empty compare: {phrases_fname} vs {gt_fname}")
            continue

        out_jsonl = os.path.join(OUT_DIR, f"{os.path.splitext(phrases_fname)[0]}__{model_id.replace('/', '__')}__judge.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as out_f:
            for j in range(n):
                img_path = image_path_for_step(action_space, cam, fps, img_prefix, j)

                res = judge.judge_pair(
                    image_path=img_path,
                    action=action_us,
                    t_idx=j,
                    cand_a=A_lines[j],
                    cand_b=B_lines[j],
                )

                rec = {
                    "phrases_file": phrases_fname,
                    "gt_file": gt_fname,
                    "action_us": action_us,
                    "action_space": action_space,
                    "action_index0": idx0,
                    "test_split_line": test_line,
                    "camera": cam,
                    "fps": fps,
                    "img_prefix": img_prefix,
                    "j": j,
                    "frame": (j + FRAME_OFFSET_LINES) * STRIDE,
                    "image_path": img_path,
                    "image_exists": os.path.exists(img_path),
                    "candidate_A_phrases": A_lines[j],
                    "candidate_B_gt": B_lines[j],
                    "winner": res.winner,
                    "confidence": res.confidence,
                    "rationale": res.rationale,
                    "raw_text": res.raw_text,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_pairs += 1

        print(f"[OK] saved: {out_jsonl} | action={action_us} idx0={idx0} cam={cam} fps={fps} compared={n}")

    print("Done.")
    print(f"missing GT matches: {missing_gt}")
    print(f"missing test_split matches: {missing_test}")
    print(f"total judged pairs: {total_pairs}")


if __name__ == "__main__":
    model_id = "OpenGVLab/InternVL2_5-2B"
    run_judge(model_id=model_id, limit_phrases_files=None)
