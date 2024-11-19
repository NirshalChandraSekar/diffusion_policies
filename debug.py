# Learning what the datasert for training a diffusion policy looks like
# For this task I am using the pusht dataset from huggingface

import torch
import torch.nn.functional as F
import numpy as np
import cv2



class ForwardProcess:
    def __init__(self, timesteps, start=0.0001, end=0.02):
        """
        Initialize the forward process with beta schedule and precompute terms.
        """
        self.timesteps = timesteps
        self.betas = self.sample_beta(start, end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def sample_beta(self, start, end, timesteps):
        """
        Generate a linear beta schedule.
        """
        return torch.linspace(start, end, timesteps)
    
    def get_index_from_list(self, values, t, shape):
        """
        Fetch a specific index t of a list of precomputed values, while respecting batch size.
        """
        batch_size = t.shape[0]
        out = values.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(t.device)
    
    def add_noise(self, image, time_step, device="cpu"):
        """
        Add noise to an image at a specific timestep t using the forward process formula.
        """
        batch_size = image.shape[0]
        t = torch.tensor([time_step] * batch_size, device=device).long()

        # Generate random noise
        noise = torch.randn_like(image)

        # Get the appropriate scaling factors for timestep t
        sqrt_alpha_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, image.shape)
        sqrt_one_minus_alpha_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, image.shape)

        # Apply the forward process formula
        noisy_image = sqrt_alpha_t * image + sqrt_one_minus_alpha_t * noise
        return noisy_image, noise




if __name__ == "__main__":

    image = cv2.imread("/home/niru/Downloads/car.jpg")
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.
    
    forward_process = ForwardProcess(500)

    noisy_image, noise = forward_process.add_noise(image, 490)
    noisy_image = noisy_image.squeeze().permute(1, 2, 0).numpy()

    cv2.imshow("Noisy Image", noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
