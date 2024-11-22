from unconditioned_diffusion import DiffusionModel
import torch
import numpy as np
import cv2

# Load the image
image = cv2.imread('/home/niru/Downloads/car.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
image = cv2.resize(image, (256, 256))  # Resize to match the model's expected size

# Normalize image to [-1, 1]
image = (image / 127.5) - 1.0  # Scale [0, 255] -> [-1, 1]

# Convert image to tensor format [1, C, H, W]
image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

# Initialize diffusion model
diffusion = DiffusionModel(image_size=64)
image = image.to(diffusion.device)

# Forward diffusion
for t in range(diffusion.time_steps):  # Loop through timesteps
    time = torch.tensor([t], device=diffusion.device)  # Time as a single integer tensor
    noisy_image, _ = diffusion.forward_diffusion(image, time)
    
    # Convert back to [0, 255] for visualization
    noisy_image_vis = (noisy_image.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
    noisy_image_vis = noisy_image_vis.clip(0, 255).astype(np.uint8)

    # Show the noisy image
    noisy_image_vis = cv2.cvtColor(noisy_image_vis, cv2.COLOR_RGB2BGR)
    cv2.imshow('Noisy Image', noisy_image_vis)
    cv2.waitKey(20)

cv2.destroyAllWindows()
