from PIL import Image, ImageDraw
import torch
import numpy as np

def generate_random_mask(img_size=(256, 256), num_rects=5, max_rect_size=(50, 50)):
    mask = Image.new("L", img_size, 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(num_rects):
        max_width, max_height = max_rect_size
        x1 = torch.randint(0, img_size[0] - max_width, (1,)).item()
        y1 = torch.randint(0, img_size[1] - max_height, (1,)).item()
        w = torch.randint(10, max_width, (1,)).item()
        h = torch.randint(10, max_height, (1,)).item()
        draw.rectangle([x1, y1, x1 + w, y1 + h], fill=255)
    return mask
