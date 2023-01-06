import torch
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.util import random_noise
import torchvision.transforms as transforms
import csv
import os

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_classes = len(classes)
n_images = 1000

classes_one_hot = F.one_hot(torch.arange(0, 10), n_classes)

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)


# define noise types
def salt_pepper_noise(image, intensity):
    image = torch.tensor(random_noise(image, mode='s&p', amount=intensity, clip=True))
    image = image.view(3, 64, 64)
    return image


def speckle_noise(image, intensity):
    image = torch.tensor(random_noise(image, mode='speckle', mean=0, var=intensity, clip=True))
    image = image.view(3, 64, 64)
    return image


def gaussian_noise(image, intensity):
    image = torch.tensor(random_noise(image, mode='gaussian', mean=0, var=intensity, clip=True))
    image = image.view(3, 64, 64)
    return image


directory = os.getcwd()

label_dir = os.path.join(directory, 'evaluation_dataset.csv')

label_list = []
img_list = []

# apply noise to test images
for i, (image, target) in enumerate(test_loader):
    if i == n_images:
        break

    gaussian = gaussian_noise(image, 0.04)
    save_image(gaussian, './gaussian (0.04)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    gaussian = gaussian_noise(image, 0.08)
    save_image(gaussian, './gaussian (0.08)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    gaussian = gaussian_noise(image, 0.16)
    save_image(gaussian, './gaussian (0.16)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    gaussian = gaussian_noise(image, 0.32)
    save_image(gaussian, './gaussian (0.32)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    gaussian = gaussian_noise(image, 0.64)
    save_image(gaussian, './gaussian (0.64)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    gaussian = gaussian_noise(image, 1.28)
    save_image(gaussian, './gaussian (1.28)/{} ({}).png'.format(target.item(), i + 1), normalize=True)


    speckle = speckle_noise(image, 0.16)
    save_image(speckle, './speckle (0.16)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    speckle = speckle_noise(image, 0.32)
    save_image(speckle, './speckle (0.32)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    speckle = speckle_noise(image, 0.64)
    save_image(speckle, './speckle (0.64)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    speckle = speckle_noise(image, 1.28)
    save_image(speckle, './speckle (1.28)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    speckle = speckle_noise(image, 2.56)
    save_image(speckle, './speckle (2.56)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    speckle = speckle_noise(image, 5.12)
    save_image(speckle, './speckle (5.12)/{} ({}).png'.format(target.item(), i + 1), normalize=True)


    snp = salt_pepper_noise(image, 0.02)
    save_image(snp, './snp (0.02)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    snp = salt_pepper_noise(image, 0.04)
    save_image(snp, './snp (0.04)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    snp = salt_pepper_noise(image, 0.08)
    save_image(snp, './snp (0.08)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    snp = salt_pepper_noise(image, 0.16)
    save_image(snp, './snp (0.16)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    snp = salt_pepper_noise(image, 0.32)
    save_image(snp, './snp (0.32)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    snp = salt_pepper_noise(image, 0.64)
    save_image(snp, './snp (0.64)/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    original = image
    save_image(original, './original/{} ({}).png'.format(target.item(), i + 1), normalize=True)

    target = int(target)
    label_one_hot = classes_one_hot[target]
    label_one_hot = label_one_hot.tolist()
    label_one_hot = ', '.join(map(str, label_one_hot))

    label_list.append(label_one_hot)
    img_list.append('{} ({}).png'.format(target, i + 1))

    list_csv = [list(i) for i in zip(img_list, label_list)]

    with open(label_dir, 'w') as file:

        write = csv.writer(file)
        write.writerows(list_csv)
