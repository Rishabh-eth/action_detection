#!/bin/bash
eval "$(/itet-stor/risingh/net_scratch/conda/bin/conda shell.bash hook)"
#eval /itet-stor/risingh/net_scratch/conda/bin/conda shell.bash hook
conda activate sfast
python tools/run_net.py --cfg /usr/itetnas04/data-scratch-01/risingh/data/videorder/experiments/joint_10_500_100_v13/SLOWFAST_32x2_R101_50_50.yaml


