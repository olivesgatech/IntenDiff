<div align="center">

<h1>Intent Makes the Difference: Intention Diffusion for Action Anticipation</h1>

<div>
    <a href='https://skvictoria.github.io' target='_blank'>Seulgi Kim</a>, Mohit Prabhushankar,
    <a href='https://alregib.ece.gatech.edu/' target='_blank'>Ghassan AlRegib</a>
</div>
<div>
    Georgia Institute of Technology (OLIVES Lab)
</div>

<div align="center">
    <h3>
        <b>Published at WACV 2027</b>
    </h3>
    <!-- <p>
        [<a href="https://arxiv.org/abs/2605.01967" target='_blank'><b>Read the Paper Here</b></a>]
    </p> -->
</div>

<div style="text-align:center">
<img src="imgs/figure1 (6).pdf" width="95%" height="100%">
</div>

---

</div>

**Abstract**

Deploying multimodal models in real-world scenarios requires generalization to new environments where recording conditions differ from training, a challenge known as multimodal domain generalization (MMDG). Standard architectures employ separate encoders for each modality and a fusion module, training the system end-to-end by optimizing on the fused features. In this paper, we identify that such joint optimization causes encoders to exploit cross-modal co-occurrences, statistical relationships between modalities that arise from source-specific recording conditions, rather than learning domain-invariant features. We term this failure mode Fusion Overfitting. To address this, we propose Modality-Entropy Regularization for Domain Generalization (MER-DG), which maximizes the entropy of each encoder's feature distribution to preserve feature diversity. MER-DG is architecture-agnostic and integrates into existing multimodal frameworks as an additive loss term. Extensive experiments on EPIC-Kitchens and HAC benchmarks demonstrate average improvements of approximately 5% over standard fusion and approximately 2% over state-of-the-art methods.


## Folder structure

```
causdiff-main/
├── src/                     # All Python code (flat imports — keep files at this level)
│   ├── main_causal_hierarchical.py  # IntenDiff entry point (--variant, --dataset)
│   ├── main.py, main-causal.py      # older baselines
│   ├── trainers*.py, models*.py, bit_diffusion*.py, batch_gen*.py, qwen*.py ...
│   ├── analysis/            #   uncertainty / attention / AUUC analysis scripts
│   ├── tools/               #   standalone utilities (qwen sub-intention extractor, npy/spectrum tools)
│   └── llm_as_a_judge/      #   VLM-as-a-judge evaluation scripts
├── scripts/
│   ├── prob/                # train/predict configs (probabilistic models)
│   ├── determ/              # train/predict configs (deterministic models)
│   └── slurm/               # sbatch launchers (run*.sh, test*.sh)
├── archive/src/             # unused experiment variants (see archive/README.md)
├── datasets/                # mappings, splits, ground truth, preprocessing scripts
│   └── darai/gt_sub_intention/   # GT sub-intentions (from llm_as_a_judge/parse_gt_sub_intention.py)
├── logs/                    # (local only, gitignored) slurm-*.out logs + INDEX.md
├── outputs/                 # (local only, gitignored)
│   ├── uncertainty_attention_out*/   # uncertainty vs. attention plots & summaries
│   └── llm_as_a_judge/               # idefics / llava / local-VLM judge results
└── notebooks/
    └── slurm_launcher.ipynb # sbatch / scancel cells
```

## Data

`datasets/*/groundTruth/` is not tracked. Download the Breakfast / 50Salads ground-truth
labels from the original dataset sources and place them under
`datasets/breakfast/groundTruth/` and `datasets/50salads/groundTruth/`.

## Running

Always run from the repo root (scripts use `./src` and `./datasets` paths):

```bash
sbatch scripts/slurm/run-causal-hierarchical.sh
```

New slurm logs go to `logs/slurm/slurm-<jobid>_<jobname>.out` automatically.
After adding logs, regenerate `logs/slurm/INDEX.md` if you want the summary up to date.

### Variants and datasets

`src/main_causal_hierarchical.py` picks the model and data loader by flag:

| `--variant` | VLM encoder | code path |
|---|---|---|
| `base` | Qwen2-VL-7B (online encoder) | `trainers_causal_hierarchical` → `bit_diffusion_causal_hierarchical_qwen` |
| `projector` | Qwen2-VL-7B, LoRA + trainable mm-projector | same, with `projector=True` |
| `lora_qwen2vl_2b` | Qwen2-VL-2B, LoRA | `trainers_causal_hierarchical_onlylora` → `bit_diffusion_causal_hierarchical_qwen_onlylora` |
| `lora_qwen2vl_7b` | Qwen2-VL-7B, LoRA, alpha-gated goal fusion | `trainers_causal_hierarchical_lora` → `bit_diffusion_causal_hierarchical_qwen_lora` |
| `lora_qwen25vl_3b` | Qwen2.5-VL-3B, LoRA | same, `vlm='qwen25vl_3b'` |
| `lora_qwen25vl_7b` | Qwen2.5-VL-7B, LoRA | same, `vlm='qwen25vl_7b'` |

- `--dataset`: `darai_l3`, `darai_l2`, `utkinects`, `nturgbd`, `breakfast`, `50salads`.
- `--goal_set`: global-goal texts (`src/goals.py`). Defaults to the one matching `--dataset`;
  set it explicitly to override (e.g. `egtea`).
- `--num_samples` defaults to 1 for `base`/`projector` and 5 otherwise.
- `--ds` is unchanged and only affects the experiment directory name.

Shared code: `bit_diffusion_common.py` (noise schedule, q-sampling, losses),
`qwen_lora_encoder.py` (LoRA VLM encoder; prompts/resolution per preset), `goals.py`.
