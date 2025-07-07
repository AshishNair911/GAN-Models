from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from utils.mask_generator import generate_random_mask

class InpaintingDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(img_path).convert("RGB")
        mask = generate_random_mask()

        image_tensor = self.transform(image)
        mask_tensor = self.transform(mask)
        masked_image = image_tensor * (1 - mask_tensor)

        input_tensor = torch.cat([masked_image, mask_tensor], dim=0)
        return input_tensor, image_tensor, mask_tensor
