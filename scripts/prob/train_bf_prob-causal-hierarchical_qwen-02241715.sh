#!/usr/bin/env bash
OPTS="--model_dir=/home/hice1/skim3513/AIFirst_F24_data/darai/models/causdiff\
      --results_dir=/home/hice1/skim3513/AIFirst_F24_data/darai/models/causdiff \
      --mapping_file=./datasets/breakfast/mapping.txt \
      --mapping_coarse_file=./datasets/breakfast/mapping_l2.txt \
      --vid_list_file_test=./datasets/breakfast/splits/test.txt \
      --vid_list_file=./datasets/breakfast/splits/train.txt \
      --gt_path=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/breakfast/groundTruth/ \
      --features_path=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/breakfast/features \
      --split=4 \
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


