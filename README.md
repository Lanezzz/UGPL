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
1.Modify the paths for the testing dataset and pre-trained model.  
2.python test_fuse.py

## Train
1.Select a certain number of ground truth, and modify the training dataset and pre-trained model paths to train the pseudo-label generator.  
python train.py  

2.Select a certain number of pseudo-labels, and modify the training dataset and pre-trained model paths to collaboratively train NS-GAN with the ground truth.  
python ST-train.py  

# Contact us
If you have any questions, please contact us (luchenyang0724@mail.dlut.edu.cn).  
