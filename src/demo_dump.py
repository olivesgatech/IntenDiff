"""
Save what src/tools/make_demo_video.py needs for one test video (opt-in via --demo_dir).

<demo_dir>/<video>_obs<obs_perc>/
    frames/0000.jpg ...   frames at the model's sample rate (as fed to the VLM)
    demo.npz              actions (S,T) sampled action ids, gt (T), obs_len,
                          subint (S,T,512) VLM-aligned sub-intentions, anchors (T,512) VLM anchors
    meta.json             video name, global intention, action names, sample rate
"""
import json
import os

import numpy as np
import torch
from einops import rearrange

from goals import goal_texts


@torch.no_grad()
def dump_demo(out_dir, args, diffusion, predictions, sample_batched, obs_perc, actions_dict):
    """predictions: (S, B, C, T) output of GaussianBitDiffusion.predict() for a batch of size 1."""
    video = str(sample_batched[-4]['file_names'][0])
    frames = sample_batched[11][0]                                   # List[PIL.Image], length T
    T = predictions.shape[-1]
    mask_past = sample_batched[5][0, 0, :T]                          # (T,)
    gt = sample_batched[1][0, :T]                                    # (T,) action ids at the sample rate
    goal_one_hot = rearrange(sample_batched[10], 'b c t -> b t c')
    goal_class = goal_one_hot[:, -1].argmax(dim=-1)                  # (1,)
    global_goal = goal_texts(args.goal_set, goal_class)

    actions = predictions.argmax(dim=2)[:, 0]                        # (S, T)
    subint = diffusion.aligned_subintentions(predictions.shape[0])[:, 0]          # (S, T, 512)
    device = next(diffusion.parameters()).device
    anchors = diffusion.vlm_anchors([frames], global_goal)[0].to(device)          # (T, 512)

    run_dir = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(video))[0]}_obs{obs_perc}")
    os.makedirs(os.path.join(run_dir, "frames"), exist_ok=True)
    for t, img in enumerate(frames[:T]):
        img.save(os.path.join(run_dir, "frames", f"{t:04d}.jpg"), quality=90)

    np.savez_compressed(
        os.path.join(run_dir, "demo.npz"),
        actions=actions.cpu().numpy().astype(np.int32),
        gt=gt.cpu().numpy().astype(np.int32),
        obs_len=int(mask_past.sum().item()),
        subint=subint.float().cpu().numpy().astype(np.float16),
        anchors=anchors.float().cpu().numpy()[:T].astype(np.float16),
    )
    id_to_action = {int(v): k for k, v in actions_dict.items()}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "video": video,
            "obs_perc": obs_perc,
            "sample_rate": args.sample_rate,
            "global_intention": global_goal if isinstance(global_goal, str) else global_goal[0],
            "action_names": id_to_action,
        }, f, indent=2)
    print(f"[demo] saved {run_dir}")
