#!/bin/sh

python scripts/train.py \
    --dataset_type=celebs_noise_reduction \
    --exp_dir=exp_name_1 \
    --workers=4 \
    --batch_size=1 \
    --test_batch_size=1 \
    --test_workers=8 \
    --val_interval=7500 \
    --save_interval=10000 \
    --encoder_type=GradualStyleEncoder \
    --start_from_latent_avg \
    --lpips_lambda=0.8 \
    --l2_lambda=1 \
    --id_lambda=0.4 \
    --optim_name=adam \
    --device=cuda \
    --noise_strength=0.5
