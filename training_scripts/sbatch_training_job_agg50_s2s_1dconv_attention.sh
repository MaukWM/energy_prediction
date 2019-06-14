#!/bin/sh
#SBATCH -J seq2seq_1dconv_attention_training
#SBATCH -c 24
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
srun python3 -m training seq2seq_1dconv_attention 50

