#!/bin/bash
#SBATCH --job-name=vi-reid
#SBATCH --nodes=1 #节点数量
#SBATCH --ntasks=8 #进程数量
#SBATCH --cpus-per-task=2 #每个进程需要调用多少cpu核数。服务器2个cpu，56核心
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G #每块gpu分配多少的内存
#SBATCH --partition=batch
#SBATCH --output=slurm-%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate python3.10
python train_sysu_baseline.py --config_file config/sysu-baseline.yml
