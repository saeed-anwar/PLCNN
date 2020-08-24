# Deep localization of protein structures in fluorescence microscopy images
This repository is for Deep localization of protein structures in fluorescence microscopy images (PLCNN) introduced in the following paper

[Muhammad Tahir], [Saeed Anwar](https://saeed-anwar.github.io/), and [Ajmal Mian] "Deep localization of protein structures in fluorescence microscopy images", [arxiv](https://arxiv.org/abs/1910.04287) 

The model is built in PyTorch 0.4.0, PyTorch 0.4.1 and tested on Ubuntu 14.04/16.04 environment (Python3.6, CUDA9.0, cuDNN5.1). 


## Contents
1. [Introduction](#introduction)
2. [Network](#network)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
Accurate localization of proteins from fluorescence microscopy images is a challenging task due to the inter-class similarities and intra-class disparities introducing grave concerns in addressing multi-class classification problems. Conventional machine learning-based image prediction relies heavily on pre-processing such as normalization and segmentation followed by hand-crafted feature extraction before classification to identify useful and informative as well as application specific features.We propose an end-to-end Protein Localization Convolutional Neural Network (PLCNN) that classifies protein localization images more accurately and reliably. PLCNN directly processes raw imagery without involving any pre-processing steps and produces outputs without any customization or parameter adjustment for a particular dataset. The output of our approach is computed from probabilities produced by the network. Experimental analysis is performed on five publicly available benchmark datasets. PLCNN consistently outperformed the existing state-of-the-art approaches from machine learning and deep architectures.

## Network

The architecture of the proposed network. A glimpse of the proposed network used for localization of the protein structures in the cell. The
composition of R_s, R_l, P_s and P_l are provided below the network structure, where the subscript s have a small number of convolutions as compared to l

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/PLCNN/blob/master/images/Network.png">
</p>

