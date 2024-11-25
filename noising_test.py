import argparse

import torch
from torchvision.utils import save_image

from ddpm import Diffusion
from utils import get_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 1  # 5
    args.img_size = 64
    args.dataset_path = r"datasets/cifar10_64/cifar10-64"
    args.train_folder = "train"
    args.val_folder = "test"
    args.slice_size = 100
    args.num_workers = 10
    args.use_wandb = False

    train_dataloader, val_dataloader = get_data(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    diff = Diffusion(device=device)

    image = next(iter(train_dataloader))[0].to(diff.device)
    t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long().to(diff.device)

    noised_image, _ = diff.noise_images(image, t)
    save_image(noised_image.add(1).mul(0.5), "noise.jpg")
