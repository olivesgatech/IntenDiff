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
| `main-causal_hierarchical_onlylora.py` | `lora_qwen2vl_2b` | `darai_l3` |
| `main-causal_hierarchical_onlylora_25b.py` | `lora_qwen25vl_3b` | `darai_l3` |
| `main-causal_hierarchical_onlylora_25b_darail2.py` | `lora_qwen25vl_3b` | `darai_l2` |
| `main-causal_hierarchical_onlylora_25b_utkinects.py` | `lora_qwen25vl_3b` | `utkinects` |
| `main-causal_hierarchical_onlylora_7b.py` | `lora_qwen2vl_7b` | `darai_l3` |
| `main-causal_hierarchical_onlylora_7b_breakfast.py` | `lora_qwen2vl_7b` | `breakfast` |
| `main-causal_hierarchical_onlylora_7b_darail2.py` | `lora_qwen2vl_7b` | `darai_l2` |
| `main-causal_hierarchical_onlylora_7b_nturgbd.py` | `lora_qwen25vl_7b` | `nturgbd` |
| `main-causal_hierarchical_onlylora_7b_utkinects.py` | `lora_qwen2vl_7b` | `utkinects` |

## Merged in step 3 (trainer / diffusion / encoder variants)

Each old file is reproduced by the new code below. For every variant, old and new
were checked to give bit-identical parameters (names, shapes, init values), training
loss, outputs, gradients and DDIM samples (with stubbed CLIP/VLM), and the encoders
to make identical model/LoRA/processor calls and prompts.

| old file | now |
|---|---|
| `trainers_causal_hierarchical_projector.py` | `trainers_causal_hierarchical.py`, `--variant=projector` |
| `trainers_causal_hierarchical_onlylora_7b.py` | `trainers_causal_hierarchical_lora.py`, `--variant=lora_qwen2vl_7b` |
| `trainers_causal_hierarchical_onlylora_25b.py` | `trainers_causal_hierarchical_lora.py`, `--variant=lora_qwen25vl_3b` |
| `trainers_causal_hierarchical_onlylora_7b_nturgbd.py` | `trainers_causal_hierarchical_lora.py`, `--variant=lora_qwen25vl_7b` |
| `bit_diffusion_causal_hierarchical_qwen_projector.py` | `bit_diffusion_causal_hierarchical_qwen.py`, `projector=True` |
| `bit_diffusion_causal_hierarchical_qwen_onlylora_7b_onlyvlm.py` | `bit_diffusion_causal_hierarchical_qwen_lora.py`, `vlm='qwen2vl_7b'` |
| `bit_diffusion_causal_hierarchical_qwen_onlylora_25b_darail2.py` | `bit_diffusion_causal_hierarchical_qwen_lora.py`, `vlm='qwen25vl_3b'` |
| `bit_diffusion_causal_hierarchical_qwen_onlylora_7b_nturgbd.py` | `bit_diffusion_causal_hierarchical_qwen_lora.py`, `vlm='qwen25vl_7b'` |
| `qwen_online_encoder_loraenable.py` | `qwen_lora_encoder.py`, `preset='qwen2vl'` |
| `qwen_2b.py` | `qwen_lora_encoder.py`, `preset='qwen2vl_2b'` |
| `qwen_25b.py` | `qwen_lora_encoder.py`, `preset='qwen25vl'` |

Old variant names: `onlylora` → `lora_qwen2vl_2b`, `onlylora_7b` → `lora_qwen2vl_7b`,
`onlylora_25b` → `lora_qwen25vl_3b`, `onlylora_7b_nturgbd` → `lora_qwen25vl_7b`.

Intentional behaviour changes:
- Goal texts now follow `--dataset` instead of being hard-coded per file. The old
  `_7b_onlyvlm` used BREAKFAST goals in training and EGTEA goals in sampling for every
  dataset; `_25b_darail2` used DARAI goals also for UT-Kinects.
- The inference-timing loop left in `_7b_nturgbd.predict()` was removed (it also
  called the sampler without `goal_one_hot`, which raises a KeyError).
- `trainers_..._onlylora_25b` / `_7b_nturgbd` hard-coded `num_samples = 5`; the
  merged trainer uses `--num_samples` (default 5 for these variants, so existing
  scripts are unchanged).
