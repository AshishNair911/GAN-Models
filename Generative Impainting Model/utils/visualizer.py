import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def visualize_output(epoch, generator, dataloader, num_images=5, device="cuda"):
    generator.eval()
    input_tensor, gt_image, mask = next(iter(dataloader))
    input_tensor, gt_image = input_tensor.to(device), gt_image.to(device)
    with torch.no_grad():
        generated_images = generator(input_tensor[:num_images])

    input_grid = make_grid(input_tensor[:num_images].cpu(), nrow=num_images, normalize=True).numpy()
    gt_grid = make_grid(gt_image[:num_images].cpu(), nrow=num_images, normalize=True).numpy()
    gen_grid = make_grid(generated_images.cpu(), nrow=num_images, normalize=True).numpy()

    def np_img(grid): return np.transpose(grid, (1, 2, 0))

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs[0].imshow(np_img(input_grid)); axs[0].set_title("Masked Input"); axs[0].axis("off")
    axs[1].imshow(np_img(gt_grid)); axs[1].set_title("Ground Truth"); axs[1].axis("off")
    axs[2].imshow(np_img(gen_grid)); axs[2].set_title("Generated Output"); axs[2].axis("off")
    plt.tight_layout(); plt.show()
