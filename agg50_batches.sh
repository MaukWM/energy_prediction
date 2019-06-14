#!/bin/sh
#SBATCH -J agg_50_training
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -c 2
#SBATCH --mem 75000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
srun python3 -m training ann 50 &
srun python3 -m training seq2seq 50 &
srun python3 -m training seq2seq_1dconv 50 &
srun python3 -m training seq2seq_attention 50 &
srun python3 -m training seq2seq_1dconv_attention 50
wait