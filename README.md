# GAN Models

This repository is dedicated to the exploration and implementation of various Generative Adversarial Network (GAN) architectures using PyTorch. The objective is to understand, replicate, and experiment with a wide range of GAN variants—from foundational models to cutting-edge advancements.

## Overview

Generative Adversarial Networks (GANs), introduced by Goodfellow et al. in 2014, have become one of the most powerful frameworks in generative modeling. This project serves as a personal and educational resource for building and understanding GAN architectures from scratch, based on academic research and practical implementations.

The repository aims to:
- Recreate foundational GAN models from key research papers
- Maintain a modular and reusable codebase for experimentation
- Document training workflows and configurations
- Share insights through structured examples and visualizations

## Table of Contents

- [Overview](#overview)
- [Implemented Architectures](#implemented-architectures)
- [Planned Architectures](#planned-architectures)
- [Setup](#setup)
- [Usage](#usage)
## Implemented Architectures

-Image Inpainting GAN (custom implementation)

This section will expand as additional GAN models are added and verified.

## Planned Architectures

- CycleGAN
- StyleGAN
- BigGAN
- Progressive Growing GAN
- Pix2Pix
- SRGAN

## Setup

Training is performed on either:
- **Kaggle T4 GPUs** (for free cloud training)
- **Local RTX 4050 Laptop GPU** with CUDA 12.8

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- tqdm
