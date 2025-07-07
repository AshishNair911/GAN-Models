import matplotlib.pyplot as plt
import numpy as np

def visualize_samples(generator, image_list, transform, generate_random_mask, input_dir, device, num_images=10):
    from PIL import Image
    import os
    import torch

    sample_images = image_list[:num_images]
    plt.figure(figsize=(15, num_images * 2))

    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(input_dir, img_name)
        image = Image.open(img_path).convert("RGB").resize((256, 256))
        mask = generate_random_mask()

        image_tensor = transform(image).to(device)
        mask_tensor = transform(mask).to(device)
        masked_image = image_tensor * (1 - mask_tensor)

        input_tensor = torch.cat([masked_image, mask_tensor], dim=0).unsqueeze(0)

        with torch.no_grad():
            output = generator(input_tensor)[0].cpu()

        original_np = image_tensor.cpu().permute(1, 2, 0).numpy()
        masked_np = masked_image.cpu().permute(1, 2, 0).numpy()
        output_np = ((output.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)

        for j, img in enumerate([original_np, masked_np, output_np]):
            ax = plt.subplot(num_images, 3, i * 3 + j + 1)
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(["Original", "Masked", "Inpainted"][j])

    plt.tight_layout()
    plt.show()
