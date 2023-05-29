#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmlab2.2
cd /home/data/xiexingxing/rcf/OBBDetection
config=$1
python tools/get_flops.py ${config} --shape 1024 1024
cd -
