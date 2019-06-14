#!/bin/sh
#SBATCH -J seq2seq_1dconv_training
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
srun python3 -m training seq2seq_1dconv 50

