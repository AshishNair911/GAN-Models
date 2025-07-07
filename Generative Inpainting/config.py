import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 50
batch_size = 8
lr = 2e-4

image_size = 256
train_data_path = "data/train/"
save_model_path = "checkpoints"
