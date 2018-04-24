#!/bin/bash
#SBATCH --gres=gpu
#SBATCH --mail-user=kdgoyette@gmail.com
#SBATCH --mail-type=ALL

source activate pykyle36
echo Running on $HOSTNAME
python pretrain_script.py
