# IntenDiff
Intent Makes the Difference: Intention Diffusion for Action Anticipation (WACV 2027)

## Folder structure

```
causdiff-main/
├── src/                     # All Python code (flat imports — keep files at this level)
│   ├── main*.py             #   entry points (called from scripts/prob, scripts/determ)
│   ├── trainers*.py, models*.py, bit_diffusion*.py, batch_gen*.py, qwen*.py ...
│   ├── analysis/            #   uncertainty / attention / AUUC analysis scripts
│   └── llm_as_a_judge/      #   VLM-as-a-judge evaluation scripts
├── scripts/
│   ├── prob/                # train/predict configs (probabilistic models)
│   ├── determ/              # train/predict configs (deterministic models)
│   └── slurm/               # sbatch launchers (run*.sh, test*.sh)
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
