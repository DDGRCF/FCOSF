#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmlab2.2
cd /home/data/xiexingxing/rcf/OBBDetection
config=$1
checkpoint=$2
cuda=$3
CUDA_VISIBLE_DEVICES=${cuda} python tools/benchmark.py ${config} ${checkpoint}
cd -
