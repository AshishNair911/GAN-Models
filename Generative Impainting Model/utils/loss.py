import torch

def wgan_generator_loss(d_fake):
    return -torch.mean(d_fake)

def wgan_discriminator_loss(d_real, d_fake):
    return torch.mean(d_fake) - torch.mean(d_real)

def compute_gradient_penalty(discriminator, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
