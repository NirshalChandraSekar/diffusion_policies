import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn 

from UNet_architecture import UNetModel
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image
import math

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

class  DiffusionModel:
    def __init__(self, time_steps=1000, beta_min=0.0001, beta_max=0.02, image_size=64, device='cuda'):

        self.time_steps = time_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.image_size = image_size
        self.device = device
        self.model = UNetModel().to(device)

        # Prepare the Beta and Alpha values for the forward and reverse diffusion process
        self.beta = torch.linspace(beta_min, beta_max, time_steps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_dash = torch.cumprod(self.alpha, 0)

        # s = 0.008
        # t = torch.linspace(0, time_steps, time_steps + 1, device=device) / time_steps
        # self.alpha_dash = torch.cos((t * (1 - s) + s) * math.pi / 2) ** 2
        # self.alpha_dash = self.alpha_dash / self.alpha_dash[0]
        # self.beta = torch.clip(1 - self.alpha_dash[1:] / self.alpha_dash[:-1], min=1e-5, max=0.999)
        # self.alpha = 1 - self.beta
        

    def prepare_data(self):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.ImageFolder("stanford_cars/cars_test", transform=transforms)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        return dataloader

    def forward_diffusion(self, images, t):
        # implement forward diffusion for the batch of images
        sqrt_alpha_dash = torch.sqrt(self.alpha_dash[t])[:, None, None, None]
        sqrt_one_minus_alpha_dash = torch.sqrt(1 - self.alpha_dash[t])[:, None, None, None]
        noise = torch.randn_like(images)
        noisy_images = sqrt_alpha_dash * images + sqrt_one_minus_alpha_dash * noise
        return noisy_images, noise
    
    def train_model(self):
        device = self.device
        dataloader = self.prepare_data()
        # model = UNetModel().to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        loss = nn.MSELoss()
        length = len(dataloader)

        for epoch in range(500):
            pbar = tqdm(dataloader)
            for i, (images, _) in enumerate(pbar):
                images = images.to(device)
                time = torch.randint(low=1, high=self.time_steps, size=(images.shape[0],)).to(device)
                noisy_images, noise = self.forward_diffusion(images, time)  
                predicted_noise = self.model(noisy_images, time)
                loss_value = loss(noise, predicted_noise)

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                pbar.set_postfix(MSE=loss_value.item())

            print(f"Epoch [{epoch + 1}/500]: Loss={loss_value.item():.4f}")
            sampled_images = self.sample_images(self.model)
            save_images(sampled_images, os.path.join("results", "unconditional_ddpm", f"{epoch}.jpg"))
            torch.save(self.model.state_dict(), os.path.join("models", "unconditional_ddpm", f"ckpt.pt"))

    def sample_images(self, model, num_samples=4):
        model.eval()
        with torch.no_grad():
            x = torch.randn((num_samples, 3, self.image_size, self.image_size)).to(self.device)
        
            # Perform reverse diffusion over timesteps
            for i in tqdm(reversed(range(1, self.time_steps)), position=0):
                t = (torch.ones(num_samples) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_dash[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                # Add noise for intermediate steps, zero noise at the last step
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                
                # Update the latent variable `x` using the reverse diffusion formula
                x = 1 / torch.sqrt(alpha) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
                ) + torch.sqrt(beta) * noise
        model.train()
        # Post-process the output to scale pixel values to [0, 255]
        x = (x.clamp(-1, 1) + 1) / 2  # Scale from [-1, 1] to [0, 1]
        x = (x * 255).type(torch.uint8)  # Scale to [0, 255] and convert to uint8
        return x


if __name__ == "__main__":
    instance = DiffusionModel(image_size=64)
    # instance.train_model()
    model = UNetModel().to(instance.device)
    ckpt = torch.load("models/unconditional_ddpm/ckpt.pt")
    model.load_state_dict(ckpt)
    image = instance.sample_images(model,  num_samples=1)
    save_images(image, os.path.join("results", "diffused_cars", "car_4.jpg"))