#!/bin/sh

#  _____ ___  ____   ___  
# |_   _/ _ \|  _ \ / _ \ 
#   | || | | | | | | | | |
#   | || |_| | |_| | |_| |
#   |_| \___/|____/ \___/  TODO: Check if we can increase batch size to 2 or 3
                        
sem -j8 python scripts/train.py --dataset_type=celebs_super_resolution --exp_dir=exp_name_1 --workers=4 --batch_size=1 --test_batch_size=1 --test_workers=8 --val_interval=7500 --save_interval=10000 --encoder_type=GradualStyleEncoder --start_from_latent_avg --lpips_lambda=0.8 --l2_lambda=1 --id_lambda=0.4 --optim_name=adam --device=cuda:0 --noise_strength=0.5 ";" echo done

sem -j8 python scripts/train.py --dataset_type=celebs_super_resolution --exp_dir=exp_name_1 --workers=4 --batch_size=1 --test_batch_size=1 --test_workers=8 --val_interval=7500 --save_interval=10000 --encoder_type=GradualStyleEncoder --start_from_latent_avg --lpips_lambda=0.8 --l2_lambda=1 --id_lambda=0.4 --optim_name=adam --device=cuda:1 --noise_strength=0.5 ";" echo done

sem -j8 ls && python scripts/train.py --dataset_type=celebs_super_resolution --exp_dir=exp_name_1 --workers=4 --batch_size=1 --test_batch_size=1 --test_workers=8 --val_interval=7500 --save_interval=10000 --encoder_type=GradualStyleEncoder --start_from_latent_avg --lpips_lambda=0.8 --l2_lambda=1 --id_lambda=0.4 --optim_name=adam --device=cuda:2 --noise_strength=0.5";" echo done

sem -j8 python scripts/train.py --dataset_type=celebs_super_resolution --exp_dir=exp_name_1 --workers=4 --batch_size=1 --test_batch_size=1 --test_workers=8 --val_interval=7500 --save_interval=10000 --encoder_type=GradualStyleEncoder --start_from_latent_avg --lpips_lambda=0.8 --l2_lambda=1 --id_lambda=0.4 --optim_name=adam --device=cuda:3 --noise_strength=0.5";" echo done

# sem --wait waits until all jobs are done.
sem --wait
