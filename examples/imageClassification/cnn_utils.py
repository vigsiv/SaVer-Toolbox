import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# restores the tensors to their original scale
def denorm(batch, mean, std):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def add_gaussian_noise(image, mean=0, std=0.1):
    noise = torch.randn(image.size()) * std + mean  # Create Gaussian noise
    noisy_image = image + noise  # Add noise to the image
    noisy_image = torch.clamp(noisy_image, 0, 1)  # Clamp to [0, 1] range
    return noisy_image

# Generate 'n' noisy samples from the same image
def generate_noisy_samples_from_image(image, n, noise_std=0.1,transform = None):
    noisy_samples = []
    for _ in range(n):
        # Add Gaussian noise to the same image
        noisy_image = add_gaussian_noise(image, std=noise_std)
        
        # Store the noisy image after required transform for nn
        if transform:
            noisy_samples.append(transform(noisy_image))
        else:
            noisy_samples.append(noisy_image)
    
    return noisy_samples

# prints image
def img_show(img,save_image = False,save_name = 'output_image.png'):
    if len(img.shape) == 4:
        simg = img.squeeze(0)  # Remove the batch dimension
    else:
        simg = img
    
    image_data = simg.permute(1, 2, 0).cpu().numpy()
    if image_data.shape[-1] == 1:
        image_data = image_data.squeeze(-1)
    plt.imshow(image_data,cmap='gray')  # Use 'gray' colormap to display a grayscale image
    plt.axis('off')
    plt.show()
    if save_image:
        plt.imsave(save_name, image_data, cmap='gray')
        

# returns logits associated with n noisy images given unnormalized image
def logit_samples(image,model,transform,n,noise_std):
    noisy_samples = generate_noisy_samples_from_image(image,n, noise_std,transform = transform)
    noisy_logits = np.array([model(s).squeeze().detach().cpu().numpy() for s in noisy_samples])
    return noisy_samples,noisy_logits