# archive

Experiment variants that no script in `scripts/` reaches any more (checked by following
active imports from every `src/*.py` entry point in `scripts/*/*.sh`).
Kept for reference / reproducing old ablations. Nothing in `src/` imports these.

To run one again, copy it back into `src/` (code uses flat imports from `src/`).

| file | what it was |
|---|---|
| `bit_diffusion_causal_hierarchical_clip.py` | goal encoder = CLIP text + `lora.py` LM conditioner |
| `bit_diffusion_causal_hierarchical_minilm.py` | goal encoder = MiniLM sentence embedding |
| `bit_diffusion_causal_hierarchical_globalgoal.py` | global-goal conditioning (no VLM) |
| `bit_diffusion_causal_hierarchical_qwen_without_vlm.py` | ablation: hierarchy without VLM encoder |
| `bit_diffusion_causal_hierarchical_qwen_onlylora_25b.py` | Qwen2.5-VL-7B, LoRA (`qwen_25b`) |
| `bit_diffusion_causal_hierarchical_qwen_onlylora_25b_freeze.py` | Qwen2.5-VL-3B, frozen VLM |
| `bit_diffusion_causal_hierarchical_qwen_onlylora_7b.py` | Qwen2-VL-7B, LoRA — superseded by `_7b_onlyvlm` |
| `bit_diffusion_causal_hierarchical_qwen_onlylora_7b_without_ft.py` | Qwen2-VL-7B, no fine-tuning |
| `qwen_online_encoder_without_ft.py` | encoder used only by `_7b_without_ft` |
| `bit_diffusion_globalgoal.py` | early non-causal global-goal diffusion |
| `bit_diffusion_onehierarchy.py` | single-level hierarchy (Qwen2.5-VL-3B) |
| `models_bit_diff_hierarchical.py` | older hierarchical TCN backbone |
| `lora.py` | LM conditioner used only by `_clip` |

## Old `main-causal_hierarchical*.py` entry points

Replaced by `src/main_causal_hierarchical.py --variant=... --dataset=...`
(verified: every script in `scripts/` produces the same trainer, batch generator,
args and tensorboard/CUDA settings as before).

| old file | `--variant` | `--dataset` |
|---|---|---|
| `main-causal_hierarchical.py` | `base` | `darai_l3` |
| `main-causal_hierarchical_projector.py` | `projector` | `darai_l3` |
| `main-causal_hierarchical_onlylora.py` | `onlylora` | `darai_l3` |
| `main-causal_hierarchical_onlylora_25b.py` | `onlylora_25b` | `darai_l3` |
| `main-causal_hierarchical_onlylora_25b_darail2.py` | `onlylora_25b` | `darai_l2` |
| `main-causal_hierarchical_onlylora_25b_utkinects.py` | `onlylora_25b` | `utkinects` |
| `main-causal_hierarchical_onlylora_7b.py` | `onlylora_7b` | `darai_l3` |
| `main-causal_hierarchical_onlylora_7b_breakfast.py` | `onlylora_7b` | `breakfast` |
| `main-causal_hierarchical_onlylora_7b_darail2.py` | `onlylora_7b` | `darai_l2` |
| `main-causal_hierarchical_onlylora_7b_nturgbd.py` | `onlylora_7b_nturgbd` | `nturgbd` |
| `main-causal_hierarchical_onlylora_7b_utkinects.py` | `onlylora_7b` | `utkinects` |
