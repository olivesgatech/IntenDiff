#!/usr/bin/env bash
# Breakfast follows the standard anticipation protocol (4 splits, results averaged over splits):
#   for SPLIT in 1 2 3 4; do SPLIT=$SPLIT bash scripts/prob/train_bf_prob-causal-hierarchical_qwen-02241715.sh; done
SPLIT=${SPLIT:-1}
OPTS="--model_dir=/home/hice1/skim3513/AIFirst_F24_data/darai/models/causdiff\
      --results_dir=/home/hice1/skim3513/AIFirst_F24_data/darai/models/causdiff \
      --mapping_file=./datasets/breakfast/mapping.txt \
      --mapping_coarse_file=./datasets/breakfast/mapping_l2.txt \
      --vid_list_file_test=./datasets/breakfast/splits/test.split${SPLIT}.bundle \
      --vid_list_file=./datasets/breakfast/splits/train.split${SPLIT}.bundle \
      --gt_path=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/breakfast/groundTruth/ \
      --features_path=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/breakfast/features \
      --split=${SPLIT} \
      --conditioned_x0 \
      --use_features \
      --use_inp_ch_dropout \
      --layer_type gated \
      --part_obs \
      --num_diff_timesteps 1000 \
      --diff_loss_type l2 \
      --diff_obj pred_x0 \
      --action=train \
      --ds=bf \
      --bz=2 \
      --lr=0.0005 \
      --model=bit-diff-pred-tcn \
      --num_epochs=100 \
      --epoch=100 \
      --num_stages=5 \
      --obs_stages=0 \
      --ant_stages=5 \
      --num_layers=9 \
      --channel_dropout_prob=0.4 \
      --num_highlevel_classes=512 \
      --date=202602241715 \
      --frames_root=/home/hice1/skim3513/AIFirst_F24_data/darai/models/causdiff/breakfast_png \
      --sample_rate=3"

python ./src/main_causal_hierarchical.py --variant=lora_qwen2vl_7b --dataset=breakfast $OPTS


