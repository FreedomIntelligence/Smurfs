#!/bin/bash
#SBATCH -A j00120230004
#SBATCH --job-name=inference_decompose
#SBATCH --gres=gpu:2
#SBATCH -p p-A800
#SBATCH -c 12

python Inference/inference_decompose.py