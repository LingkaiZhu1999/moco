#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=20
#SBATCH --gpus=4
#SBATCH --partition=gpu-v100-16g

module load scicomp-python-env

python main_moco.py \
  "/scratch/shareddata/dldata/imagenet-1k-wds/imagenet-1k-wds/" \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --workers 10 \
  --dist-url 'tcp://localhost:10001'\
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --dist-backend 'nccl' \
  --compile \
  --resume /scratch/work/zhul2/code/moco/saved_model/resnet50/mocov1/checkpoint_0190.pth.tar
