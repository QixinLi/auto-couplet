#!/bin/bash
#SBATCH -J yklai
#SBATCH -o ./outputs/0426-roberta-5e-5
#SBATCH -p compute
#SBATCH -N 1                         
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1

export PYTHONPATH=..

conda activate cuda

python train.py --debug
