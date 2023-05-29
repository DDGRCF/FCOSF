<div align="center"> 

<h1>FCOSF🚀</h1> 

**Offical Implementation for Learning Orientation-aware Distances for Oriented Object Detection**

</div>

## Introduction

This is the official implementation of [FCOSF](https://ieeexplore.ieee.org/document/10130561), which is implemented on [OBBDetection](https://github.com/jbwang1997/OBBDetection)

## Update

- (**2023-05-29**) Release [FCOSF](configs/fcosf/fcosf_r50.py)🔥.
- (**2023-05-29**) Release [MMRotate Implementation](git@github.com:DDGRCF/FCOSF-MMRotate.git) (stay tuning)🙍.

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Get Started

### How to use OBBDetection

If you want to train or test a oriented model, please refer to [oriented_model_starting.md](docs/oriented_model_starting.md).

### How to Start FCOSF

#### Train

To help you start quickly, I prepare a simple bash script

```bash
config=/path/to/config && work_dir=/path/to/work_dir && cuda=(device_id, like 1, 2, 3 ...)
bash my_exps/train.sh ${config} ${work_dir} ${cuda}
```

#### Test

```bash
config=/path/to/config && ckpt=/path/to/checkpoint && save_dir=/path/to/results_save_dir && cuda=(same as above)
bash my_exps/test.sh ${config} ${ckpt} ${save_dir} ${cuda}
```

### How to Deploy the FCOSF

TODO:

## Cite

```shell
@article{fcosf,
  author={Rao, Chaofan and Wang, Jiabao and Cheng, Gong and Xie, Xingxing and Han, Junwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Learning Orientation-aware Distances for Oriented Object Detection}, 
  year={2023},
  pages={1-1},
  doi={10.1109/TGRS.2023.3278933}}
```

## License
This project is released under the [Apache 2.0 license](LICENSE).
