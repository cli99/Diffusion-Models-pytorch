import torch
from fastcore.all import *
from torchvision.utils import save_image

from ddpm_conditional import *

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

n = 10
diff = Diffusion(
    img_size=config.img_size, num_classes=config.num_classes, device=device
)
diff.load("artifacts/model:v33")
labels = torch.ones(10).long().to(diff.device)
samples = diff.sample(use_ema=False, labels=labels)
save_images(samples, "samples.jpg")
