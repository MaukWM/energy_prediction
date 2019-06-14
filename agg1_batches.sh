#!/bin/sh
#SBATCH -J agg_1_training
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -c 2
#SBATCH --mem 75000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
srun python3 -m training ann 1 &
srun python3 -m training seq2seq 1 &
srun python3 -m training seq2seq_1dconv 1 &
srun python3 -m training seq2seq_attention 1 &
srun python3 -m training seq2seq_1dconv_attention 1
wait