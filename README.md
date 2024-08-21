This forked project integrates the Progressive Enhancement Learning (PEL) framework from the paper [Exploiting Fine-grained Face Forgery Clues via Progressive Enhancement Learning](https://arxiv.org/abs/2112.13977) into a synthetic speech detection system. By using LFCC and CQT as features, we enhance the detection of subtle forgery clues in synthetic speech. The implementation includes self-enhancement and mutual-enhancement modules to progressively refine feature learning, inspired by the techniques used in the PEL framework for face forgery detection.

***
---
# Implementation of Progressive Enhancement Learning for Audio Deepfake Detection

This repository contains an implementation of Progressive Enhancement Learning applied to the detection of audio deepfakes.

## Directory Structure

- **Progressive_Enhancement_Learning/**: This directory houses the primary scripts for this project.

    - `demo1.py`: The original implementation of the [makaijie/End-to-End-Dual-Branch-Network-Towards-Synthetic-Speech-Detection](https://github.com/makaijie/End-to-End-Dual-Branch-Network-Towards-Synthetic-Speech-Detection) repository.

    - `demo2.py`: An implementation incorporating Self and Mutual Enhancement based on the [paper](https://arxiv.org/pdf/2112.13977).

## Current Status

- The code has been tested on single audio samples.
- The full training pipeline will be updated soon.

---
# End-to-End Dual-Branch Network Towards Synthetic Speech Detection  

### Prerequisites

- NVIDIA GPU+CUDA CuDNN
- Install Torch1.8 and dependencies

### Training and Test Details
- Please adjust the file location before training and testing;
- Data Preparation
  - Change the `Feature Engineering/CQT/cqt_extract.py`, `Feature Engineering/LFCC/extract_lfcc.m` and `Feature Engineering/LFCC/reload_data.py`
  - Run the `Feature Engineering/CQT/cqt_extract.py`, `Feature Engineering/LFCC/extract_lfcc.m` and `Feature Engineering/LFCC/reload_data.py`

- When you train the network
  - Change the `dual-branch_sum_loss.py` or `dual-branch_alternative_loss.py`
  - Run the `dual-branch_sum_loss.py` or `dual-branch_alternative_loss.py`

- When you test the network 
  - Change the `Result_sum_loss/test_dual.py` or `Result_alternative_loss/test_dual.py`
  - Run the `Result_sum_loss/test_dual.py` or `Result_alternative_loss/test_dual.py`

### Acknowledgements
The code of this work is adapted from https://github.com/yzyouzhang/AIR-ASVspoof, https://github.com/yzyouzhang/Empirical-Channel-CM and https://github.com/joaomonteirof/e2e_antispoofing.
