#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -c 2
#SBATCH --mem-per-cpu=3G
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH --output=log-%j.out
#SBATCH -e log-%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fb38
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python collect_pose_dataset.py --furniture_name lamp 
