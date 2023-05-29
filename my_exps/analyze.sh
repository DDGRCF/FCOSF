#! /bin/bash
env=obbdetn
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${env}
task=$1
log_json=$2
options=${@:3}

python tools/analyze_logs.py ${task} ${log_json} ${options}
