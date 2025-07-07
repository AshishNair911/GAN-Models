import torch
import torch.nn.functional as F

def wgan_generator_loss(d_fake):
    return -torch.mean(d_fake)

def wgan_discriminator_loss(d_real, d_fake):
    return torch.mean(d_fake) - torch.mean(d_real)

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = (epsilon * real_samples + (1 - epsilon) * fake_samples).requires_grad_(True)

    d_interpolated = D(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty
