#! /bin/bash
:<<!
  A Simple Shell Script, that help you start quickly!
!

# activate your python env
env=your_env # note: reset your env
source /path/to/your/anaconda3/etc/profile.d/conda.sh # note: reset your env

if [ $? -ne 0 ]; then
  echo -e "\033[31msource conda path fail! please check your setting\033[0m"
  exit 1
fi
conda activate ${env}
if [ $? -ne 0 ]; then
  echo -e "\033[31msource conda path fail! please check your setting\033[0m"
  exit 1
fi

echo -e "\033[34m*******************************\033[0m"
echo -e "\033[32mactivate env ${env}\033[0m"
echo -e "\033[34m*******************************\033[0m"
echo -e "\033[32mcurrent dir is ${PWD}\033[0m"
echo -e "\033[34m*******************************\033[0m"

task=$1
log_json=$2
options=${@:3}

python tools/analyze_logs.py ${task} ${log_json} ${options}
