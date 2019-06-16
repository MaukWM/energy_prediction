#!/bin/sh
#SBATCH -J ann_training_50
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
srun python3 -m training ann 50

