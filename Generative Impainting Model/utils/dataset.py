import os
from torch.utils.data import Dataset
from PIL import Image
from .mask_utils import generate_random_mask
from torchvision import transforms

class InpaintingDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(image_path).convert("RGB").resize((256, 256))
        mask = generate_random_mask((256, 256))

        image = self.transform(image)
        mask_tensor = self.transform(mask)
        masked_image = image * (1 - mask_tensor)
        input_tensor = torch.cat([masked_image, mask_tensor], dim=0)
        return input_tensor, image, mask_tensor
