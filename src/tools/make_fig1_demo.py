"""
Build an *illustrative* demo folder from Fig. 1 of the paper, for make_demo_video.py.

Everything comes from Fig. 1: the four photos (a), the sub-intention phrases (a), and the
GTDA / IntenDiff action samples and ground truth (c), with segment lengths read off the figure.
The uncertainty curves are hand-set to tell the figure's story; they are NOT model outputs,
and the rendered video says so.

    python src/tools/make_fig1_demo.py imgs/figure1.pdf /tmp/fig1_demo
    python src/tools/make_demo_video.py /tmp/fig1_demo -o imgs/intendiff-demo.gif --fps 5
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

T, OBS = 40, 27   # time steps; the observation ends where Fig. 1 draws the cut (~68%)

# photo boxes in the 2782x939 raster embedded in figure1.pdf (number tags cropped off)
BOXES = [(43, 82, 457, 300), (461, 82, 875, 300), (879, 82, 1293, 300), (1302, 82, 1706, 300)]
SPANS = [(0, 9), (9, 18), (18, OBS), (OBS, T)]   # time steps shown by each photo

CAPTIONS = [
    "A person is standing in front of a coffee machine, possibly preparing to make coffee.",
    "A person is holding a cup, possibly will use a coffee maker located nearby.",
    "A person is opening a coffee powder packet and appears to be about to pour it.",
    None,
]

ACTIONS = ["Get_a_cup", "Get_coffee", "Pour_coffee_powder", "Add_water",
           "Add_flour", "Load_dishwasher", "Mix_pancake_ingredients"]
GROUP = {0: "Making coffee", 1: "Making coffee", 2: "Making coffee", 3: "Making coffee",
         4: "Making a pancake", 5: "Cleaning dishes", 6: "Making a pancake"}
# Fig. 1 colour coding: pink = making coffee, blue = making a pancake, yellow = cleaning dishes
GROUP_COLORS = {"Making coffee": "#e2689b", "Making a pancake": "#2a78d6", "Cleaning dishes": "#eda100"}
# steps of "making coffee" get light -> dark pink shades, in task order (as in Fig. 1)
ACTION_COLORS = {0: "#fde2ec", 1: "#f4a6c5", 2: "#e2689b", 3: "#bd3a70"}

# (action id, end step) runs, read off Fig. 1(c)
ROWS = {
    "GTDA · sample 1":      [(0, 9), (1, 22), (2, 27), (3, 40)],
    "GTDA · sample 2":      [(1, 11), (4, 21), (5, 40)],
    "GTDA · sample 3":      [(0, 14), (6, 40)],
    "IntenDiff · sample 1": [(1, 23), (2, 27), (3, 40)],
    "IntenDiff · sample 2": [(0, 40)],
    "IntenDiff · sample 3": [(0, 8), (1, 21), (2, 27), (3, 40)],
}
GT = [(0, 17), (1, 27), (3, 40)]


def runs_to_seq(runs):
    seq, start = np.zeros(T, dtype=np.int32), 0
    for a, end in runs:
        seq[start:end] = a
        start = end
    return seq


def ramp(*knots):
    """piecewise-linear curve through (step, value) knots"""
    xs, ys = zip(*knots)
    return np.interp(np.arange(T), xs, ys).round(3).tolist()


def load_raster(pdf_or_png):
    if pdf_or_png.endswith(".png"):
        return Image.open(pdf_or_png).convert("RGB")
    tmp = tempfile.mkdtemp()
    subprocess.run(["pdfimages", "-png", "-f", "1", "-l", "1", pdf_or_png, os.path.join(tmp, "fig")], check=True)
    return Image.open(os.path.join(tmp, "fig-000.png")).convert("RGB")


def ken_burns(img, n, zoom=0.10, out_size=(560, 296)):
    """slow zoom-in on a still photo so it reads as video"""
    w, h = img.size
    frames = []
    for k in range(n):
        z = 1 + zoom * k / max(n - 1, 1)
        cw, ch = w / z, h / z
        x0, y0 = (w - cw) / 2, (h - ch) * 0.4
        frames.append(img.crop((x0, y0, x0 + cw, y0 + ch)).resize(out_size, Image.LANCZOS))
    return frames


def main():
    src, out = sys.argv[1], sys.argv[2]
    raster = load_raster(src)
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    t = 0
    for box, (a, b) in zip(BOXES, SPANS):
        for frame in ken_burns(raster.crop(box), b - a):
            frame.save(os.path.join(out, "frames", f"{t:04d}.jpg"), quality=92)
            t += 1

    actions = np.stack([runs_to_seq(r) for r in ROWS.values()])
    np.savez_compressed(os.path.join(out, "demo.npz"), actions=actions, gt=runs_to_seq(GT), obs_len=OBS)

    captions = [None] * T
    for (a, b), c in zip(SPANS, CAPTIONS):
        captions[a:b] = [c] * (b - a)
    meta = {
        "video": "Fig. 1 (illustration)",
        "global_intention": "Making coffee",
        "action_names": {str(i): n for i, n in enumerate(ACTIONS)},
        "action_groups": {str(i): g for i, g in GROUP.items()},
        "group_colors": GROUP_COLORS,
        "action_colors": {str(i): c for i, c in ACTION_COLORS.items()},
        "legend_title": "Colour = global intention the action belongs to (light → dark pink: steps of making coffee)",
        "row_labels": list(ROWS),
        "captions": captions,
        "banner": "Illustration of Fig. 1 · not model output",
        # GTDA vs IntenDiff, using the sample rows above. Observation-level uncertainty is computed
        # from those rows (entropy across each method's samples). Intention-level uncertainty (Eq. 2)
        # needs sub-intentions, which GTDA does not have; IntenDiff's curve is hand-set to tell the
        # figure's story (ambiguity shrinks over frames 1-3, grows moderately into the unseen future).
        "compare": {
            "methods": ["GTDA", "IntenDiff"],
            "rows": {"GTDA": [0, 1, 2], "IntenDiff": [3, 4, 5]},
            "colors": {"GTDA": "#8a8984", "IntenDiff": "#6250d6"},
            "intention": {"IntenDiff": ramp((0, 0.62), (8, 0.56), (9, 0.50), (17, 0.40), (18, 0.32), (26, 0.24), (39, 0.42))},
            "missing_text": "not modeled (no sub-intentions)",
            "drift_group": "Making coffee",
            "note": "intention: illustrative values • observation: entropy across the 3 samples below\n"
                    "off-intention: the sampled action belongs to another global intention",
        },
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
