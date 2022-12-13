# Crowd Estimation
This repository contains several crowd estimation algorithms to apply my task.
I strongly reuse the implementations of `Tracking Pedestrian Heads in Dense Crowd(CVPR 2021)` and 
`An end to end transformer model for crowd localization(ECCV 2022)`

## Dataset

1. For HeadHunter--T
- CroHD dataset  [here](https://motchallenge.net/data/Head_Tracking_21/). 
2. For CLTR
- JHU-Crowd++ dataset [here](http://www.crowd-counting.com/)
- NWPU-Crowd dataset [here](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)


## Setup Instructions

### 1. HeadHunter--T

In order to execute this codebase the following requirements need to be satisfied. 

- Nvidia Driver >= 418
- Cuda 10.0 is needed if Docker is unavailable.
- head_detection folder from HeadHunter [here](https://github.com/Sentient07/HeadHunter) 
- Python packages : To install the required python packages;
	```conda env create -f env.yml```

### 2. CLTR
* Please change ```nproc_per_node``` and ``` gpu_id``` of ```jhu.sh/nwpu.sh```, if you do not have enogh GPU.
* We have fixed all random seeds, i.e., different runs will report the same results under the same setting.
* The model will be saved in ```CLTR/save_file/log_file```
* Note that using FPN will improve the performance, but we do not add it in this version.
* Turning some hyper-parameters will also bring improvement (e.g., the image size, crop size, number of queries).


## Instructions to Run
```
run video_demo.py # with your arguments
```



## Reference :

```
@InProceedings{Sundararaman_2021_CVPR,
    author    = {Sundararaman, Ramana and De Almeida Braga, Cedric and Marchand, Eric and Pettre, Julien},
    title     = {Tracking Pedestrian Heads in Dense Crowd},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3865-3875}
}
```
```
@article{liang2022end,
  title={An end-to-end transformer model for crowd localization},
  author={Liang, Dingkang and Xu, Wei and Bai, Xiang},
  journal={European Conference on Computer Vision},
  year={2022}

```

