#!/bin/bash
#SBATCH -o slurm/adam-%a-%j.out
#SBATCH -e slurm/adam-%a-%j.err
#SBATCH --mem=10G
#SBATCH -p scavenger-gpu --gres=gpu:1 
#SBATCH -c 6
module unload Python
module load Python-GPU/3.7.6
python adam_exp.py
