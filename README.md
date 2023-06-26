# Know Your Space: Inlier and Outlier Construction for Calibrating Medical OOD Detectors

![](https://img.shields.io/badge/pytorch-green)
This repository hosts the official PyTorch implementation for: [`Know Your Space: Inlier and Outlier Construction for Calibrating Medical OOD Detectors`](https://openreview.net/forum?id=RU7fr0-M8N), MIDL 2023.

![Functional Diagram](teaser.PNG)

## Dependencies
This codebase was developed and tested using

+ matplotlib `3.4.3`
+ numpy `1.20.3`
+ scikit_learn `1.1.3`
+ torch `1.10.0`
+ opencv `0.6`

## Downloading MedMNIST Datasets, Train/Val Splits and Model Checkpoints
[Click here](https://arizonastateu-my.sharepoint.com/:u:/g/personal/vnaray29_sundevils_asu_edu/ESSQ986FmfdPlBNtnYLVD9AB6cAZFjWVUeuD0kW28ltslQ?e=e2vj3A) to download the MedMNIST datasets along with the Train/Val Split CSV Files. Extract them in your working directory.

## 1. Training the classifier with Latent Space Inliers and Pixel Space Outliers  
To train the classifier on a dataset from MedMNIST (for e.g., bloodmnist)

```
python train_in_latent_out_pix.py --in_dataset bloodmnist --model_type wrn --dist --augmix --rand_conv --ckpt_name in_latent_out_pix
```

## 2. Perform OOD Detection

```
python ood_detection.py --in_dataset bloodmnist --model_type wrn --dir_name in_latent_out_pix
```

NOTE: For performing OOD detection, you must have the classifier trained on the given dataset stored in `./ckpts`

# To reproduce our results
Download the checkpoints from the link provided above. Extract the .zip file as is in your working directory. Then execute
```
python ood_detection.py --in_dataset bloodmnist --model_type wrn --dir_name in_latent_out_pix
```

## Citation

Our paper can be cited as:

```
@inproceedings{narayanaswamy2023know,
title={Know Your Space: Inlier and Outlier Construction for Calibrating Medical {OOD} Detectors},
author={Vivek Narayanaswamy and Yamen Mubarka and Rushil Anirudh and Deepta Rajan and Andreas Spanias and Jayaraman J. Thiagarajan},
booktitle={Medical Imaging with Deep Learning},
year={2023}}
```

## Acknowledgments

We adapt the official implementation of Virtual Outlier Synthesis for implementing our baselines and algorithms: https://github.com/deeplearning-wisc/vos. We sincerely thank the authors for open-sourcing their code.

We thank the authors of ['MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification'](https://medmnist.com/) for providing the MedMNIST Benchmark

We thank the authors of ['Augmix'](https://github.com/google-research/augmix) and ['RandConv'](https://github.com/wildphoton/RandConv) for open-sourcing their implementations

We thank the authors of ['Robust Out-of-distribution Detection in Neural Networks'](https://github.com/jfc43/robust-ood-detection) for open sourcing their code base.

## License
This code is distributed under the terms of the MIT license. All new contributions must be made under this license. LLNL-CODE-850636 SPDX-License-Identifier: MIT
