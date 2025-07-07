import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model.generator import AOTGenerator
from model.discriminator import PatchDiscriminator
from utils.dataset import InpaintingDataset
from utils.loss import wgan_generator_loss, wgan_discriminator_loss, compute_gradient_penalty
from utils.visualizer import visualize_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_dir = "data/train/img"
batch_size = 24
num_epochs = 100
lr = 1e-5
n_critic = 5
lambda_gp = 10
alpha = 100  # weight for L1 loss

# Dataset and Dataloader
dataset = InpaintingDataset(image_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Models
generator = AOTGenerator().to(device)
discriminator = PatchDiscriminator().to(device)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss
l1_loss_fn = nn.L1Loss()

# Training Loop
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()

    epoch_loss_d = 0
    epoch_loss_g = 0

    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for input_tensor, gt_image, _ in pbar:
            input_tensor = input_tensor.to(device)
            gt_image = gt_image.to(device)

            # === Train Discriminator ===
            for _ in range(n_critic):
                fake_images = generator(input_tensor)

                d_real = discriminator(gt_image)
                d_fake = discriminator(fake_images.detach())

                d_loss = wgan_discriminator_loss(d_real, d_fake)
                gp = compute_gradient_penalty(discriminator, gt_image, fake_images.detach(), device)

                d_total_loss = d_loss + lambda_gp * gp

                d_optimizer.zero_grad()
                d_total_loss.backward()
                d_optimizer.step()

            # === Train Generator ===
            fake_images = generator(input_tensor)
            d_fake = discriminator(fake_images)

            g_adv_loss = wgan_generator_loss(d_fake)
            g_l1_loss = l1_loss_fn(fake_images, gt_image)
            g_total_loss = g_adv_loss + alpha * g_l1_loss

            g_optimizer.zero_grad()
            g_total_loss.backward()
            g_optimizer.step()

            epoch_loss_d += d_total_loss.item()
            epoch_loss_g += g_total_loss.item()
            pbar.set_postfix(D_loss=d_total_loss.item(), G_loss=g_total_loss.item(), G_L1=g_l1_loss.item())

    print(f"Epoch [{epoch+1}/{num_epochs}] | D: {epoch_loss_d/len(dataloader):.4f}, G: {epoch_loss_g/len(dataloader):.4f}")
    
    # Visualization & Checkpoint
    visualize_output(epoch, generator, dataloader, num_images=5, device=device)
    torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch+1}.pth")
    torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch+1}.pth")
    print(f"Saved models for epoch {epoch+1}")
