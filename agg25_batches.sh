#!/bin/sh
#SBATCH -J agg_25_training
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -c 2
#SBATCH --mem 75000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
srun python3 -m training ann 25 &
srun python3 -m training seq2seq 25 &
srun python3 -m training seq2seq_1dconv 25 &
srun python3 -m training seq2seq_attention 25 &
srun python3 -m training seq2seq_1dconv_attention 25
wait