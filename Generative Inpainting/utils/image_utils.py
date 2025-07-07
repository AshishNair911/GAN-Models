import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np

def save_tensor_image(tensor, path):
    if isinstance(tensor, torch.Tensor):
        grid = vutils.make_grid(tensor, normalize=True)
        ndarr = grid.mul(255).add(0.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(path)
