# Deep localization of protein structures in fluorescence microscopy images
This repository is for Deep localization of protein structures in fluorescence microscopy images (PLCNN) introduced in the following paper

[Muhammad Tahir], [Saeed Anwar](https://saeed-anwar.github.io/), and [Ajmal Mian] "Deep localization of protein structures in fluorescence microscopy images", [arxiv](https://arxiv.org/abs/1910.04287) 

The model is built in PyTorch 0.4.0, PyTorch 0.4.1 and tested on Ubuntu 14.04/16.04 environment (Python3.6, CUDA9.0, cuDNN5.1). 


## Contents
1. [Introduction](#introduction)
2. [Network](#network)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
Accurate localization of proteins from fluorescence microscopy images is a challenging task due to the inter-class similarities and intra-class disparities introducing grave concerns in addressing multi-class classification problems. Conventional machine learning-based image prediction relies heavily on pre-processing such as normalization and segmentation followed by hand-crafted feature extraction before classification to identify useful and informative as well as application specific features.We propose an end-to-end Protein Localization Convolutional Neural Network (PLCNN) that classifies protein localization images more accurately and reliably. PLCNN directly processes raw imagery without involving any pre-processing steps and produces outputs without any customization or parameter adjustment for a particular dataset. The output of our approach is computed from probabilities produced by the network. Experimental analysis is performed on five publicly available benchmark datasets. PLCNN consistently outperformed the existing state-of-the-art approaches from machine learning and deep architectures.

<p align="center">
  <img width="700" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/Example_images.png">
</p>
Image datasets for protein localization; each image belongs to a different class. Most of the images
are sparse.

## Network

The architecture of the proposed network. A glimpse of the proposed network used for localization of the protein structures in the cell. The
composition of R_s, R_l, P_s and P_l are provided below the network structure, where the subscript s have a small number of convolutions as compared to l

<p align="center">
  <img width="700" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/Network.png">
</p>

## Test
### Quick start
1. Download the trained models for our paper and place them in '/TestCode/experiment'.

    The PLCNN model can be downloaded from [Google Drive]() or [here](). The total size for all models is 5MB.

2. Cd to '/TestCode/code', run the following scripts.

    **You can use the following script to test the algorithm**

    ```bash
    #RIDNET
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model RIDNET --n_feats 64 --pre_train ../experiment/ridnet.pt --test_only --save_results --save 'RIDNET_RNI15' --testpath ../LR/LRBI/ --testset RNI15
    ```


## Results
**All the results for RIDNET can be downloaded from GoogleDrive from [SSID](https://drive.google.com/open?id=15peD5EvQ5eQmd-YOtEZLd9_D4oQwWT9e), [RNI15](https://drive.google.com/open?id=1PqLHY6okpD8BRU5mig0wrg-Xhx3i-16C) and [DnD](https://noise.visinf.tu-darmstadt.de/submission-detail). The size of the results is 65MB** 

### Quantitative Results
<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/CNNvsMachine.png">
</p>
The performance of state-of-the-art algorithms on widely used publicly available DnD dataset in terms of PSNR (in dB) and SSIM. The best results are highlighted in bold.

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/PLCNNvsTCNN.PNG">
</p>
The quantitative results (in PSNR (dB)) for the SSID and Nam datasets.. The best results are presented in bold.

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/EffectDecreasingTraining.PNG">
</p>

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/ETAS_accuracy.PNG">
</p>


For more information, please refer to our [papar](https://arxiv.org/abs/1904.07396)

### Confusion matrices 
A real noisy example from DND dataset for comparison of our method against the state-of-the-art algorithms.

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/CHO_CM.PNG">
</p>

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/DeepYeast_CM.PNG">
</p>

Comparison on more samples from DnD. The sharpness of the edges on the objects and textures restored by our method is the best.

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/Correctpredict.PNG">
</p>

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/PLCNNvsTCNNGraph.PNG">
</p>


![Visual_PSNR_SSIM_BI](/images/CAM1.PNG)
![Visual_PSNR_SSIM_BI](/images/CAM2.PNG)
Few more examples from SSID dataset.

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{tahir2019PLCNN,
  title={Deep localization of protein structures in fluorescence microscopy images},
  author={Tahir, Muhammad and Anwar, Saeed and Mian, Ajmal},
  journal={arXiv preprint arXiv:1910.04287},
  year={2019}
}

```
