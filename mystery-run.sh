#!/usr/bin/env bash

# Source bashrc
#source $HOME/ .bashrc

#source activate sai

sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 launch_script.py
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 launch_script.py
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 launch_script.py
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 launch_script.py
