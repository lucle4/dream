import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.util import random_noise
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


n_images = 10


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=n_images, shuffle=True)


# define noise types
def salt_pepper_noise(image):
    image = torch.tensor(random_noise(image, mode='s&p', salt_vs_pepper=0.5, clip=True))
    image = image.view(3, 64, 64)
    return image

def speckle_noise(image):
    image = torch.tensor(random_noise(image, mode='speckle', mean=0, var=0.05, clip=True))
    image = image.view(3, 64, 64)
    return image

def gaussian_noise(image):
    image = torch.tensor(random_noise(image, mode='gaussian', mean=0, var=0.05, clip=True))
    image = image.view(3, 64, 64)
    return image


# apply noise to test images
for i, (image, target) in enumerate(test_loader):
    if i == n_images:
        break

    else:
        snp = salt_pepper_noise(image[i])
        save_image(snp, './salt and pepper/{} ({}).png'.format(target[i], i+1), normalize=True)

        speckle = speckle_noise(image[i])
        save_image(speckle, './speckle/{} ({}).png'.format(target[i], i+1), normalize=True)

        gaussian = gaussian_noise(image[i])
        save_image(gaussian, './gaussian/{} ({}).png'.format(target[i], i+1), normalize=True)
