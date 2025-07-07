import matplotlib.pyplot as plt
import numpy as np
import torch

def tensor_to_img(tensor):
    """Convert [-1, 1] or [0, 1] Tensor image to NumPy image [0, 1]"""
    tensor = tensor.detach().cpu()
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2  # [-1,1] → [0,1]
    return tensor.clamp(0, 1).permute(1, 2, 0).numpy()

def visualize_output(epoch, generator, dataloader, num_images=5, device="cuda"):
    generator.eval()
    input_tensor, gt_image, mask = next(iter(dataloader))
    input_tensor = input_tensor.to(device)
    gt_image = gt_image.to(device)

    with torch.no_grad():
        generated_images = generator(input_tensor[:num_images])

    plt.figure(figsize=(12, num_images * 3))

    for i in range(num_images):
        orig = tensor_to_img(gt_image[i])
        masked = tensor_to_img(input_tensor[i, :3] * (1 - input_tensor[i, 3:4]))
        output = tensor_to_img(generated_images[i])

        for j, img in enumerate([orig, masked, output]):
            ax = plt.subplot(num_images, 3, i * 3 + j + 1)
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(["Original", "Masked", "Inpainted"][j])

    plt.suptitle(f"Inpainting Results (Epoch {epoch})", fontsize=16)
    plt.tight_layout()
    plt.show()
