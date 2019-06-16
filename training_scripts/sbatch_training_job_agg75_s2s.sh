#!/bin/sh
#SBATCH -J seq2seq_training_75
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
srun python3 -m training seq2seq 75

