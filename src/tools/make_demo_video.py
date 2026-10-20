"""
Render a README demo (GIF / MP4) from a folder written by src/demo_dump.py.

    python src/tools/make_demo_video.py <demo_dir>/<video>_obs0.3 -o imgs/demo.gif [--mp4 imgs/demo.mp4]

Layout: the video plays on the left (frames after the observation cut are dimmed: the model
never sees them); top right shows the two uncertainties at the current time; the bottom shows
the S sampled action sequences and the ground truth, revealed as the video plays.

  intention-level uncertainty  U_l = E_s || P_l P_l^T (hatI_l^(s) - mu_l) ||^2   (paper Eq. 2;
                                P_l = top-m right singular vectors of the residuals)
  observation-level uncertainty = entropy of the S sampled actions at time l / log S   (as in Fig. 5)
"""
import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

# reference categorical palette (fixed order) + roles, light surface
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#b5b4ae"
SURFACE = "#fcfcfb"
TEXT, TEXT_2, MUTED, GRID, TRACK = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ecebe7"
INTENT_HUE, OBS_HUE = "#2a78d6", "#eb6834"   # sequential contexts: blue, then orange


def intention_uncertainty(subint, anchors, m):
    """subint (S,T,D), anchors (T,D) -> (T,)"""
    S, T, _ = subint.shape
    out = np.zeros(T)
    for l in range(T):
        r = subint[:, l] - anchors[l]                         # residuals (S,D)
        _, _, vt = np.linalg.svd(r, full_matrices=False)
        basis = vt[:min(m, vt.shape[0])].T                    # (D,m) orthonormal tangent basis
        coef = np.einsum("sd,dm->sm", r, basis)               # ||P P^T r||^2 == ||P^T r||^2
        out[l] = np.mean(np.sum(coef ** 2, axis=1))
    return out


def observation_uncertainty(actions):
    """actions (S,T) sampled ids -> (T,) normalised entropy over samples"""
    S, T = actions.shape
    out = np.zeros(T)
    for l in range(T):
        _, counts = np.unique(actions[:, l], return_counts=True)
        p = counts / S
        out[l] = -(p * np.log(p)).sum() / np.log(S) if S > 1 else 0.0
    return out


def segments(seq):
    """run-length encode -> [(start, end, label)]"""
    out, start = [], 0
    for t in range(1, len(seq) + 1):
        if t == len(seq) or seq[t] != seq[start]:
            out.append((start, t, int(seq[start])))
            start = t
    return out


def color_map(gt, actions, names, ignore):
    """slot colours by first appearance (GT first); past 8 actions fold into Other"""
    order = []
    for seq in [gt] + list(actions):
        for a in seq:
            a = int(a)
            if a not in order and a not in ignore and str(a) in names:
                order.append(a)
    cmap = {a: SERIES[i] for i, a in enumerate(order[:len(SERIES)])}
    return cmap, order


def pretty(name):
    return name.replace("_", " ")


class GifWriter(animation.AbstractMovieWriter):
    """GIF with a median-cut palette per frame: keeps flat UI colours exact and photos smooth
    (matplotlib's PillowWriter shares one quantisation that shifts light grays toward the series hues)."""

    def __init__(self, fps, hold_last=0, fixed_colors=()):
        super().__init__(fps=fps)
        self.hold_last, self.frames = hold_last, []
        # exact UI colours go into every palette; small marks (legend chips) would otherwise be merged away
        self.fixed = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in dict.fromkeys(fixed_colors)]

    def setup(self, fig, outfile, dpi=None):
        self.fig, self.outfile = fig, outfile

    def grab_frame(self, **savefig_kwargs):
        self.fig.canvas.draw()
        rgb = Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy())
        adaptive = rgb.quantize(colors=256 - len(self.fixed), method=Image.Quantize.MEDIANCUT).getpalette()
        flat = [c for rgb3 in self.fixed for c in rgb3] + adaptive[:3 * (256 - len(self.fixed))]
        pal = Image.new("P", (1, 1))
        pal.putpalette(flat + [0] * (768 - len(flat)))
        self.frames.append(rgb.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG))

    def finish(self):
        ms = int(round(1000 / self.fps))
        durations = [ms] * len(self.frames)
        durations[-1] += ms * self.hold_last
        self.frames[0].save(self.outfile, save_all=True, append_images=self.frames[1:], duration=durations, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", help="folder written by demo_dump.py")
    ap.add_argument("-o", "--out", default="demo.gif")
    ap.add_argument("--mp4", default=None, help="also write an MP4 (needs ffmpeg)")
    ap.add_argument("--tangent_dim", type=int, default=2, help="m in Eq. 2")
    ap.add_argument("--fps", type=float, default=6)
    ap.add_argument("--hold", type=int, default=12, help="extra frames to hold the last frame")
    ap.add_argument("--max_samples", type=int, default=5, help="sample rows to draw")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    d = np.load(os.path.join(args.demo, "demo.npz"))
    meta = json.load(open(os.path.join(args.demo, "meta.json")))
    names = meta["action_names"]
    # optional meta keys
    groups = meta.get("action_groups")      # {action id: group}: colour by group (e.g. global intention, as in Fig. 1)
    group_colors = meta.get("group_colors", {})
    row_labels = meta.get("row_labels")     # labels for the sample rows; "A · x" rows are grouped by "A"
    captions = meta.get("captions")         # per-step sub-intention text shown under the frame
    banner = meta.get("banner")
    actions, gt, obs_len = d["actions"], d["gt"], int(d["obs_len"])
    T = actions.shape[1]
    valid = gt < len(names)                                    # drop padding at the end
    T = int(valid.sum()) if valid.any() else T
    actions, gt = actions[:, :T], gt[:T]

    if "uncertainty" in meta:                                  # precomputed curves
        u_int = np.asarray(meta["uncertainty"]["intention"], dtype=float)[:T]
        u_obs = np.asarray(meta["uncertainty"]["observation"], dtype=float)[:T]
    else:
        subint, anchors = d["subint"][:, :T].astype(np.float64), d["anchors"][:T].astype(np.float64)
        u_int = intention_uncertainty(subint, anchors, args.tangent_dim)
        u_obs = observation_uncertainty(actions)
    u_int_n = u_int / max(u_int.max(), 1e-12)

    frame_dir = os.path.join(args.demo, "frames")
    frames = [Image.open(os.path.join(frame_dir, f"{t:04d}.jpg")).convert("RGB") for t in range(T)]
    n_rows = len(row_labels) if row_labels else args.max_samples
    rows = actions[:n_rows]
    S = rows.shape[0]
    if groups:
        order = list(dict.fromkeys(groups[str(int(a))] for seq in [gt] + list(rows) for a in seq))
        slot = {g: group_colors.get(g, SERIES[k % len(SERIES)]) for k, g in enumerate(order)}
        cmap = {int(a): slot[g] for a, g in groups.items()}
    else:
        cmap, order = color_map(gt, rows, names, ignore=set())

    # ---------------- figure ----------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.facecolor": SURFACE})
    tall = S > 5
    fig = plt.figure(figsize=(12.8, 8.0 if tall else 7.2), dpi=80, facecolor=SURFACE)
    tl_top = 0.47 if tall else 0.42
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1], left=0.035, right=0.975,
                  top=0.9, bottom=tl_top + 0.08, wspace=0.08)
    gs_tl = GridSpec(1, 1, figure=fig, left=0.15 if row_labels else 0.11, right=0.975, top=tl_top, bottom=0.1)
    title = args.title or f"Global intention: “{meta['global_intention']}”"
    fig.text(0.035, 0.95, title, fontsize=17, color=TEXT, weight="bold", va="center")
    if banner:
        fig.text(0.975, 0.95, banner, fontsize=11, color=MUTED, ha="right", va="center", style="italic")

    ax_vid = fig.add_subplot(gs[0, 0])
    ax_vid.axis("off")
    im = ax_vid.imshow(np.asarray(frames[0]))
    tag = ax_vid.text(0.015, 0.97, "", transform=ax_vid.transAxes, va="top", fontsize=12, color=TEXT,
                      bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec="none", alpha=0.92))
    cap = ax_vid.text(0.5, -0.03, "", transform=ax_vid.transAxes, ha="center", va="top", fontsize=12,
                      color=TEXT_2, style="italic", linespacing=1.3)

    # uncertainty panel: two separate measures -> two bars + two sparklines (never one dual axis)
    right = gs[0, 1].subgridspec(4, 1, height_ratios=[0.55, 1, 0.55, 1], hspace=0.55)
    panels = []
    for k, (label, hue, series, raw) in enumerate([
            ("Intention-level uncertainty", INTENT_HUE, u_int_n, u_int),
            ("Observation-level uncertainty", OBS_HUE, u_obs, u_obs)]):
        ax_bar = fig.add_subplot(right[2 * k])
        ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1); ax_bar.axis("off")
        ax_bar.add_patch(Rectangle((0, 0.05), 1, 0.45, fc=TRACK, ec="none"))
        fill = ax_bar.add_patch(Rectangle((0, 0.05), 0, 0.45, fc=hue, ec="none"))
        ax_bar.text(0, 0.78, label, fontsize=12, color=TEXT, va="center")
        val = ax_bar.text(1, 0.78, "", fontsize=12, color=TEXT_2, va="center", ha="right")
        ax_sp = fig.add_subplot(right[2 * k + 1])
        ax_sp.set_xlim(0, T - 1); ax_sp.set_ylim(0, 1.08)
        for s in ax_sp.spines.values():
            s.set_visible(False)
        ax_sp.tick_params(colors=MUTED, labelsize=9, length=0)
        ax_sp.set_yticks([0, 1]); ax_sp.set_xticks([])
        ax_sp.axhline(0, color=GRID, lw=1)
        ax_sp.axvline(obs_len - 0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
        line, = ax_sp.plot([], [], color=hue, lw=2, solid_capstyle="round")
        dot, = ax_sp.plot([], [], "o", ms=8, color=hue, mec=SURFACE, mew=2)
        panels.append((fill, val, line, dot, series, raw))
    note = meta.get("uncertainty_note") or ("intention: Eq. 2, bar scaled to this video's max • observation: "
                                            f"entropy across {actions.shape[0]} action samples")
    ax_sp.set_xlabel(note, fontsize=9, color=MUTED, loc="right", labelpad=4)

    # action timelines (Fig. 1c style)
    labels = list(row_labels) if row_labels else [f"Sample {i + 1}" for i in range(S)]
    ys, y = [], 0.0
    for i, lab in enumerate(labels):
        if i and lab.split(" · ")[0] != labels[i - 1].split(" · ")[0]:
            y += 0.5                                           # gap between row groups (e.g. GTDA / IntenDiff)
        ys.append(y); y += 1
    ys.append(y + 0.6); labels.append("Ground truth")
    y_max = ys[-1] + 1.0
    ax_tl = fig.add_subplot(gs_tl[0])
    ax_tl.set_xlim(0, T); ax_tl.set_ylim(-0.6, y_max)
    ax_tl.invert_yaxis()
    for s in ax_tl.spines.values():
        s.set_visible(False)
    ax_tl.set_xticks([]); ax_tl.tick_params(length=0, labelsize=11, colors=TEXT_2)
    ax_tl.set_yticks(ys); ax_tl.set_yticklabels(labels)
    for y, seq in zip(ys, list(rows) + [gt]):
        for a, b, lab in segments(seq):
            c = cmap.get(lab, OTHER)
            ax_tl.add_patch(Rectangle((a, y - 0.38), b - a, 0.76, fc=c, ec=SURFACE, lw=2))
            text = pretty(names.get(str(lab), "?"))
            if (b - a) / T * 1000 >= 9 * len(text) + 12:          # label only when it fits
                ax_tl.text((a + b) / 2, y, text, ha="center", va="center", fontsize=10, color=TEXT, clip_on=True)
    ax_tl.axvline(obs_len, color=TEXT, lw=2)
    ax_tl.text(obs_len / 2, y_max - 0.15, "← Partial observation →", ha="center", va="center", color=TEXT_2, fontsize=11)
    ax_tl.text((obs_len + T) / 2, y_max - 0.15, "← Anticipation →", ha="center", va="center", color=TEXT_2, fontsize=11)
    cover = ax_tl.add_patch(Rectangle((0, -0.6), T, y_max + 0.6, fc=SURFACE, ec="none", alpha=0.82, zorder=3))
    head = ax_tl.axvline(0, color=TEXT, lw=1, ls=(0, (2, 2)), zorder=4)

    # legend (identity is never colour alone: segments are labelled too)
    if groups:
        handles = [Rectangle((0, 0), 1, 1, fc=slot[g]) for g in order]
        lab_txt = list(order)
    else:
        shown = order[:len(SERIES)]
        handles = [Rectangle((0, 0), 1, 1, fc=cmap[a]) for a in shown]
        lab_txt = [pretty(names[str(a)]) for a in shown]
        if len(order) > len(SERIES):
            handles.append(Rectangle((0, 0), 1, 1, fc=OTHER)); lab_txt.append("Other")
    fig.legend(handles, lab_txt, loc="lower center", ncol=min(len(handles), 6), frameon=False,
               fontsize=10, labelcolor=TEXT_2, handlelength=1.2, bbox_to_anchor=(0.5, 0.0),
               title=meta.get("legend_title"), title_fontsize=10)

    n_frames = T

    def draw(i):
        t = min(i, T - 1)
        frame = np.asarray(frames[t]).astype(np.float32)
        if t >= obs_len:  # unseen by the model: dim towards the surface
            frame = frame * 0.45 + 252 * 0.55
            tag.set_text(f"Anticipation • not seen by model   t = {t + 1}/{T}")
        else:
            tag.set_text(f"Observed   t = {t + 1}/{T}")
        im.set_data(frame.astype(np.uint8))
        c = captions[t] if captions else None
        cap.set_text(textwrap.fill(f"Sub-intention: “{c}”", 70) if c else "")
        for fill, val, line, dot, series, raw in panels:
            fill.set_width(float(series[t]))
            val.set_text(f"{raw[t]:.2f}")
            line.set_data(np.arange(t + 1), series[:t + 1])
            dot.set_data([t], [series[t]])
        cover.set_x(t + 1); cover.set_width(max(T - t - 1, 0))
        head.set_xdata([t + 0.5, t + 0.5])
        return []

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, interval=1000 / args.fps, blit=False)
    fills = list(cmap.values()) + [OTHER, INTENT_HUE, OBS_HUE]
    covered = ["#%02x%02x%02x" % tuple(int(round(int(h[i:i + 2], 16) * 0.18 + int(SURFACE[i:i + 2], 16) * 0.82))
                                       for i in (1, 3, 5)) for h in fills]   # fills seen through the 0.82 cover
    ui = [SURFACE, TEXT, TEXT_2, MUTED, GRID, TRACK, "#ffffff"] + fills + covered
    anim.save(args.out, writer=GifWriter(args.fps, hold_last=args.hold, fixed_colors=ui),
              savefig_kwargs={"facecolor": SURFACE})
    print("wrote", args.out)
    if args.mp4:
        anim.save(args.mp4, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400),
                  savefig_kwargs={"facecolor": SURFACE})
        print("wrote", args.mp4)


if __name__ == "__main__":
    main()
