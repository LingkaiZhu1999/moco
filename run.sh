#!/bin/bash
#SBATCH --time=60:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-gpu=40
#SBATCH --gpus=2
#SBATCH --partition=gpu-h200-141g-ellis

module load scicomp-python-env
# source /scratch/work/zhul2/code/assignment1-basics/.venv/bin/activate
# module load triton/2025.1-gcc
# module load cuda
# python main.py \
#   "/scratch/shareddata/dldata/imagenet-1k-wds/imagenet-1k-wds/" \
#   --arch resnet50 \
#   --batch-size 256 \
#   --workers 20 \
#   --dist-url 'tcp://127.0.0.1:10000'\
#   --dist-backend 'nccl'\
#   --world-size 1 \
#   --rank 0 \
#   --multiprocessing-distributed \
#   --compile

TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1 python main_moco.py \
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
  --compile
