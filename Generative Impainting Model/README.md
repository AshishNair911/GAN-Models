# Video Inpainting using Generative Adversarial Networks

This project implements a full pipeline for **video frame inpainting** using a GAN-based model trained on occluded dashcam or surveillance-like footage. It supports training from scratch, evaluation, and real-world inference on corrupted or masked videos.

---

## Highlights

- Custom generator (AOTGenerator with AOTBlocks) and PatchDiscriminator.
- WGAN-GP training with additional L1 loss.
- Automatic black-region mask detection for inference.
- Full pipeline: video → frames → masked input → inpainting → output video.
- Dataset-independent: works on test sets or new videos.

---

## Model Overview

### Generator: `AOTGenerator`

- Encoder-decoder architecture with downsampling and `UpConv` modules.
- Uses two `AOTBlock`s with multi-dilation rates (`[1, 2, 4, 8]`).
- Outputs 3-channel RGB images with `Tanh` activation.

### Discriminator: `PatchDiscriminator`

- Multi-layer convolutional architecture.
- Patch-based output for local realism feedback.

---

## Losses

- **WGAN-GP** Loss
- **Gradient Penalty**
- **L1 Loss** for pixel-wise accuracy
- Final generator loss:  
  `G_loss = -D(G(x)) + α * L1(G(x), real)`, where α = 100

---

## Directory Structure

