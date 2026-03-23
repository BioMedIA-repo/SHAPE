# SHAPE

This is the official implementation
of [SHAPE: Structure-aware Hierarchical Unsupervised Domain Adaptation with Plausibility Evaluation for Medical Image Segmentation]
at CVPR-2026.

## Table of Contents

- [Requirements](#requirements)
- [Download](#download)
- [Train](#train)

## Requirements

Run the following command to install the required packages:

```bash
conda create --name new_env --file environment.txt
```

## Download

You can download the dinov3 pre-trained models
from [here](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).

## Train

### 1. Dataset Preparation

Please organise the dataset according to the following structure,where the npz file stores the images and their
corresponding segmentation labels with the key name {image, label}:

```angular2
data/
└── data_name/
    ├── train/
    │   ├── 00001.npz
    │   └── ...
    ├── val/
    │   ├── 00010.npz
    │   └── ...
    └── test/
        ├── 00020.npz
        └── ...
```

### 2. Training

Now you can start to train the model:

```angular2
python train.py --mode <CT/MR/...> --gpu <gpu id> --stage <sup/unsup> --dino_size <b/s> --checkpoint_name <exp_name> --use_hfm --use_pseudo_labels --use_selector --use_refinement
```

For MMWHS dataset with CT labeled:

```angular2
python train.py --mode CT --gpu <gpu id> --stage <sup/unsup> --dino_size <b/s> --checkpoint_name <exp_name> --use_hfm --use_pseudo_labels --use_selector --use_refinement
```

For MMWHS dataset with MR labeled:

```angular2
python train.py --mode MR --gpu <gpu id> --stage <sup/unsup> --dino_size <b/s> --checkpoint_name <exp_name> --use_hfm --use_pseudo_labels --use_selector --use_refinement
```

For abdominal dataset with CT labeled:

```angular2
python train.py --mode ABCT --gpu <gpu id> --stage <sup/unsup> --dino_size <b/s> --checkpoint_name <exp_name> --use_hfm --use_pseudo_labels --use_selector --use_refinement
```

For abdominal dataset with MR labeled:

```angular2
python train.py --mode ABMR --gpu <gpu id> --stage <sup/unsup> --dino_size <b/s> --checkpoint_name <exp_name> --use_hfm --use_pseudo_labels --use_selector --use_refinement
```

## Acknowledgement

The project is based on [dinov3](https://github.com/facebookresearch/dinov3).
We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

## Citations

If the code is helpful for your research, please consider citing:

```angular2
@inproceedings{zhou2026shape,
  author={Linkuan Zhou, Yinghao Xia, Yufei Shen, Xiangyu Li, Wenjie Du, Cong Cong, Leyi Wei, Ran Su, Qiangguo Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  title={SHAPE: Structure-aware Hierarchical Unsupervised Domain Adaptation with Plausibility Evaluation for Medical Image Segmentation}, 
  year={2026},
  pages={1-11}
}
```

## Social media

<p align="center"><img width="600" alt="image" src="https://github.com/BioMedIA-repo/.github/blob/052046a248d3831a599e11c85ff94cdd658c5abc/pic/wechat.png" height=""></p> 
Welcome to follow our [Wechat official account: iBioMedInfo] and [Xiaohongshu official account: iBioMedInfo], we will share recent studies on biomedical image and bioinformation analysis there.

## Global Collaboration & Questions

**Global Collaboration:** We're on a mission to biomedical research, aiming for artificial intelligence and its
applications to biomedical image and bioinformation analysis, promoting the development of the medical community.
Collaborate with us to increase competitiveness.

**Questions:** General questions, please contact 'zlinkw@mail.nwpu.edu.cn'
