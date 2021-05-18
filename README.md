# DRO-LT: Distributional Robustness Loss for Long-tail Learning
[Paper](https://arxiv.org/abs/2104.03066)  
[Project Website](https://chechiklab.biu.ac.il/~dvirsamuel/DROLT/)  
[Video](https://www.youtube.com/watch?v=iFt2w9wPsw4)

## Overview
DRO-LT loss is a new loss based on robustness theory, which encourages the model to learn high-quality representations for both head and tail classes. DRO-LT reduces representation bias towards head classes in the feature space and increases recognition accuracy of tail classes while  largely maintaining the accuracy of head classes. Our new robustness loss can be combined with various classifier balancing techniques and can be applied to representations at several layers of the deep model.

## Requirements

Quick installation under Anaconda:
```
conda env create -f requirements.yml
```

## Training with DRO-LT
**1.** Train a vanilla Cross-Entropy model on CIFAR-10-LT/CIFAR-100 LT
```
PYTHONPATH="./" python main.py --gpu 1 --dataset cifar100 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None
```
Model checkpoints will be saved in directory inside the project directory.
Code is based on [LDAM-DRW](https://github.com/kaidic/LDAM-DRW).

**2.** Continue training with DRO-LT
```
PYTHONPATH="./" python main.py --gpu 1 --dataset cifar100 --imb_type exp --imb_factor 0.01 --resume True --epochs 100 --pretrained cifar100_resnet32_CE_None_exp_0.01_0 --feat_sampler resample,4 --feat_lr 0.005 --feat_loss ce_lt,robust_loss --cls_sampler none --cls_lr 0.01 --cls_loss ce --temperature 1 -b 128 --margin 1 --margin_type learned
```
**--epochs:** Number of epochs for training with DRO-LT loss  
**--pretrained:** Path to pretrained cross-entropy model    
**--feat_sampler:** Sample type for the feature extractor part of the model. (type, args)  
**--feat_lr:** Feature extractor learning rate  
**--feat_loss:** Loss types for training the feature extractor part  
**--cls_sampler:** Sampler type for the classifier part  
**--cls_lr:** Classifier learning rate  
**--cls_loss:** Loss types for training the classifier  
**--temperature:** DRO-LT loss temperature  
**--b:** Batch Size  
**--margin_type:** Type of ball margin (learned = "Learned epsilon").  

Model checkpoints will be saved inside the CE model folder from step (1).

**3.** Evaluate the trained model
To evaluate the model add ```--evaluation True```
```
PYTHONPATH="./" python main.py --gpu 1 --dataset cifar100 --imb_type exp --imb_factor 0.01 --resume True --epochs 100 --pretrained cifar100_resnet32_CE_None_exp_0.01_0 --feat_sampler resample,4 --feat_lr 0.005 --feat_loss ce_lt,robust_loss --cls_sampler none --cls_lr 0.01 --cls_loss ce --temperature 1 -b 128 --margin 1 --margin_type learned --evaluation True
```

## Pretrained CIFAR-100-LT model
A. Download from [this link](https://chechiklab.biu.ac.il/~dvirsamuel/DROLT/models/cifar100_resnet32_CE_None_exp_0.01_0.zip)  
B. Unzip the folder and place it under the working directory (under DRO-LT folder)

## Cite Our Paper
If you find our paper and repo useful, please cite:
```
@InProceedings{Samuel_2021_ICCV,
    author    = {Samuel, Dvir and Chechik, Gal},
    title     = {Distributional Robustness Loss for Long-Tail Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021},
```