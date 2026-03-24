#!/bin/bash
#SBATCH -t 0:20:00
#SBATCH -c 1
#SBATCH --mem-per-cpu=5G
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH --output=log-%j.out
#SBATCH -e log-%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fb38
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python train_val_networks_fb.py --pc &
python train_val_networks_fb.py &
wait
