#!/bin/sh
#SBATCH -J agg_75_training
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -c 2
#SBATCH --mem 75000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
srun python3 -m training ann 75 &
srun python3 -m training seq2seq 75 &
srun python3 -m training seq2seq_1dconv 75 &
srun python3 -m training seq2seq_attention 75 &
srun python3 -m training seq2seq_1dconv_attention 75
wait