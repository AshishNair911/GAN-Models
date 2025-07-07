import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model.generator import AOTGenerator
from utils.mask_utils import generate_random_mask
from utils.video_utils import video_to_frames, frames_to_video

# === Paths ===
video_path = "input_video.mp4"
frame_input_folder = "frames_input"
frame_output_folder = "frames_output"
output_video_path = "inpainted_output.mp4"
model_path = "checkpoints/generator_epoch_100.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
generator = AOTGenerator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# === Video to Frames ===
video_to_frames(video_path, frame_input_folder)

# === Inpaint Each Frame ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

os.makedirs(frame_output_folder, exist_ok=True)

with torch.no_grad():
    for fname in tqdm(sorted(os.listdir(frame_input_folder)), desc="Inpainting frames"):
        if not fname.endswith(".png"):
            continue
        img_path = os.path.join(frame_input_folder, fname)
        image = Image.open(img_path).convert("RGB").resize((256, 256))
        mask = generate_random_mask((256, 256))

        image_tensor = transform(image).to(device)
        mask_tensor = transform(mask).to(device)
        masked_image = image_tensor * (1 - mask_tensor)

        input_tensor = torch.cat([masked_image, mask_tensor], dim=0).unsqueeze(0)

        output = generator(input_tensor.to(device))[0].cpu()
        output_img = transforms.ToPILImage()(output.clamp(-1, 1) * 0.5 + 0.5)  # [-1, 1] → [0, 1]
        output_img.save(os.path.join(frame_output_folder, fname))

# === Reconstruct Video ===
frames_to_video(frame_output_folder, output_video_path)
print("Inference completed and video saved.")
