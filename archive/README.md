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
