# UGPL
Accepted paper in Neurips2022:
Semi-Supervised Video Salient Object Detection Based on Uncertainty-Guided Pseudo Labels
Yongri Piao, Chenyang Lu, Miao Zhang, Huchuan Lu.

# Prerequisites
- Ubuntu 20.04
- CUDA 11.3
- PyTorch 1.7.0
- Python 3.6

[pretrained models](https://pan.baidu.com/s/1hEts3xx_pwY-Fejmepj0SQ) ,code：zve9
# Train/Test
## Test
Modify the paths for the testing dataset and pre-trained model(10GT+50PL_best.pth).  
- python test_fuse.py

## Train
1.Select a certain number of ground truth, and modify the training dataset and pre-trained model(pretrain_resnet50.pth) paths to train the pseudo-label generator.  
- python train.py  

You can also use our pretrained model (pseudo_label.pth) to generate pseudo-labels.

2.Select a certain number of pseudo-labels, and modify the training dataset and pre-trained model paths(pretrain_resnet50.pth for RGB stream & resnet50-19c8e357.pth for OPT stream) to collaboratively train NS-GAN with the ground truth.  
- python ST-train.py  

# Contact us
If you have any questions, please contact us (luchenyang0724@mail.dlut.edu.cn).  
