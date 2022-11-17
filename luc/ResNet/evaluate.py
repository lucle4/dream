import os
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from torchvision.io import read_image
import torchvision.models as models
import torch.nn as nn
from torchvision.models import resnet50
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

directory = os.getcwd()

label_dir = os.path.join(directory, 'evaluation/evaluation_dataset.csv')

img_dir_original = os.path.join(directory, 'evaluation/original')

img_dir_gaussian_002 = os.path.join(directory, 'evaluation/gaussian (0.02)')
img_dir_gaussian_004 = os.path.join(directory, 'evaluation/gaussian (0.04)')
img_dir_gaussian_008 = os.path.join(directory, 'evaluation/gaussian (0.08)')
img_dir_gaussian_016 = os.path.join(directory, 'evaluation/gaussian (0.16)')
img_dir_gaussian_032 = os.path.join(directory, 'evaluation/gaussian (0.32)')
img_dir_gaussian_064 = os.path.join(directory, 'evaluation/gaussian (0.64)')

img_dir_speckle_004 = os.path.join(directory, 'evaluation/speckle (0.04)')
img_dir_speckle_008 = os.path.join(directory, 'evaluation/speckle (0.08)')
img_dir_speckle_016 = os.path.join(directory, 'evaluation/speckle (0.16)')
img_dir_speckle_032 = os.path.join(directory, 'evaluation/speckle (0.32)')
img_dir_speckle_064 = os.path.join(directory, 'evaluation/speckle (0.64)')
img_dir_speckle_128 = os.path.join(directory, 'evaluation/speckle (0.128)')

img_dir_snp_002 = os.path.join(directory, 'evaluation/salt and pepper (0.02)')
img_dir_snp_004 = os.path.join(directory, 'evaluation/salt and pepper (0.04)')
img_dir_snp_008 = os.path.join(directory, 'evaluation/salt and pepper (0.08)')
img_dir_snp_016 = os.path.join(directory, 'evaluation/salt and pepper (0.16)')
img_dir_snp_032 = os.path.join(directory, 'evaluation/salt and pepper (0.32)')
img_dir_snp_064 = os.path.join(directory, 'evaluation/salt and pepper (0.64)')


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ConvertImageDtype(float),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class Dataset(Dataset):
    def __init__(self, label_dir, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(label_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        label = [float(i) for i in label.split(',')]
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


model_50k_0k = resnet50(weights=None)
model_50k_0k.fc = nn.Linear(2048, 10)

model_40k_10k = resnet50(weights=None)
model_40k_10k.fc = nn.Linear(2048, 10)

model_30k_20k = resnet50(weights=None)
model_30k_20k.fc = nn.Linear(2048, 10)

model_20k_30k = resnet50(weights=None)
model_20k_30k.fc = nn.Linear(2048, 10)


if torch.cuda.is_available():
    model_50k_0k.cuda()
    model_40k_10k.cuda()
    model_30k_20k.cuda()


model_50k_0k.load_state_dict(torch.load(os.path.join(directory, 'checkpoints_50k_0k/checkpoint epoch 100.pt')))
model_50k_0k.eval()

model_40k_10k.load_state_dict(torch.load(os.path.join(directory, 'checkpoints_40k_10k/checkpoint epoch 100.pt')))
model_40k_10k.eval()

model_30k_20k.load_state_dict(torch.load(os.path.join(directory, 'checkpoints_30k_20k/checkpoint epoch 100.pt')))
model_30k_20k.eval()

model_20k_30k.load_state_dict(torch.load(os.path.join(directory, 'checkpoints_20k_30k/checkpoint epoch 100.pt')))
model_20k_30k.eval()


running_accuracy_original_50k_0k = 0.0
running_accuracy_original_40k_10k = 0.0
running_accuracy_original_30k_20k = 0.0
running_accuracy_original_20k_30k = 0.0


running_accuracy_gaussian_002_50k_0k = 0.0
running_accuracy_gaussian_002_40k_10k = 0.0
running_accuracy_gaussian_002_30k_20k = 0.0
running_accuracy_gaussian_002_20k_30k = 0.0

running_accuracy_gaussian_004_50k_0k = 0.0
running_accuracy_gaussian_004_40k_10k = 0.0
running_accuracy_gaussian_004_30k_20k = 0.0
running_accuracy_gaussian_004_20k_30k = 0.0

running_accuracy_gaussian_008_50k_0k = 0.0
running_accuracy_gaussian_008_40k_10k = 0.0
running_accuracy_gaussian_008_30k_20k = 0.0
running_accuracy_gaussian_008_20k_30k = 0.0

running_accuracy_gaussian_016_50k_0k = 0.0
running_accuracy_gaussian_016_40k_10k = 0.0
running_accuracy_gaussian_016_30k_20k = 0.0
running_accuracy_gaussian_016_20k_30k = 0.0

running_accuracy_gaussian_032_50k_0k = 0.0
running_accuracy_gaussian_032_40k_10k = 0.0
running_accuracy_gaussian_032_30k_20k = 0.0
running_accuracy_gaussian_032_20k_30k = 0.0

running_accuracy_gaussian_064_50k_0k = 0.0
running_accuracy_gaussian_064_40k_10k = 0.0
running_accuracy_gaussian_064_30k_20k = 0.0
running_accuracy_gaussian_064_20k_30k = 0.0


running_accuracy_speckle_004_50k_0k = 0.0
running_accuracy_speckle_004_40k_10k = 0.0
running_accuracy_speckle_004_30k_20k = 0.0
running_accuracy_speckle_004_20k_30k = 0.0

running_accuracy_speckle_008_50k_0k = 0.0
running_accuracy_speckle_008_40k_10k = 0.0
running_accuracy_speckle_008_30k_20k = 0.0
running_accuracy_speckle_008_20k_30k = 0.0

running_accuracy_speckle_016_50k_0k = 0.0
running_accuracy_speckle_016_40k_10k = 0.0
running_accuracy_speckle_016_30k_20k = 0.0
running_accuracy_speckle_016_20k_30k = 0.0

running_accuracy_speckle_032_50k_0k = 0.0
running_accuracy_speckle_032_40k_10k = 0.0
running_accuracy_speckle_032_30k_20k = 0.0
running_accuracy_speckle_032_20k_30k = 0.0

running_accuracy_speckle_064_50k_0k = 0.0
running_accuracy_speckle_064_40k_10k = 0.0
running_accuracy_speckle_064_30k_20k = 0.0
running_accuracy_speckle_064_20k_30k = 0.0

running_accuracy_speckle_128_50k_0k = 0.0
running_accuracy_speckle_128_40k_10k = 0.0
running_accuracy_speckle_128_30k_20k = 0.0
running_accuracy_speckle_128_20k_30k = 0.0


running_accuracy_snp_002_50k_0k = 0.0
running_accuracy_snp_002_40k_10k = 0.0
running_accuracy_snp_002_30k_20k = 0.0
running_accuracy_snp_002_20k_30k = 0.0

running_accuracy_snp_004_50k_0k = 0.0
running_accuracy_snp_004_40k_10k = 0.0
running_accuracy_snp_004_30k_20k = 0.0
running_accuracy_snp_004_20k_30k = 0.0

running_accuracy_snp_008_50k_0k = 0.0
running_accuracy_snp_008_40k_10k = 0.0
running_accuracy_snp_008_30k_20k = 0.0
running_accuracy_snp_008_20k_30k = 0.0

running_accuracy_snp_016_50k_0k = 0.0
running_accuracy_snp_016_40k_10k = 0.0
running_accuracy_snp_016_30k_20k = 0.0
running_accuracy_snp_016_20k_30k = 0.0

running_accuracy_snp_32_50k_0k = 0.0
running_accuracy_snp_32_40k_10k = 0.0
running_accuracy_snp_32_30k_20k = 0.0
running_accuracy_snp_32_20k_30k = 0.0

running_accuracy_snp_64_50k_0k = 0.0
running_accuracy_snp_64_40k_10k = 0.0
running_accuracy_snp_64_30k_20k = 0.0
running_accuracy_snp_64_20k_30k = 0.0

total = 0


original_dataset = Dataset(label_dir, img_dir_original, transform=transform)
training_size = len(original_dataset)

original_loader = DataLoader(original_dataset, batch_size=training_size, shuffle=True)


gaussian_002_dataset = Dataset(label_dir, img_dir_gaussian_002, transform=transform)
gaussian_004_dataset = Dataset(label_dir, img_dir_gaussian_004, transform=transform)
gaussian_008_dataset = Dataset(label_dir, img_dir_gaussian_008, transform=transform)
gaussian_016_dataset = Dataset(label_dir, img_dir_gaussian_016, transform=transform)
gaussian_032_dataset = Dataset(label_dir, img_dir_gaussian_032, transform=transform)
gaussian_064_dataset = Dataset(label_dir, img_dir_gaussian_064, transform=transform)

gaussian_002_loader = DataLoader(gaussian_002_dataset, batch_size=1, shuffle=True)
gaussian_004_loader = DataLoader(gaussian_004_dataset, batch_size=training_size, shuffle=True)
gaussian_008_loader = DataLoader(gaussian_008_dataset, batch_size=training_size, shuffle=True)
gaussian_016_loader = DataLoader(gaussian_016_dataset, batch_size=1, shuffle=True)
gaussian_032_loader = DataLoader(gaussian_032_dataset, batch_size=training_size, shuffle=True)
gaussian_064_loader = DataLoader(gaussian_064_dataset, batch_size=training_size, shuffle=True)


speckle_004_dataset = Dataset(label_dir, img_dir_speckle_004, transform=transform)
speckle_008_dataset = Dataset(label_dir, img_dir_speckle_008, transform=transform)
speckle_016_dataset = Dataset(label_dir, img_dir_speckle_016, transform=transform)
speckle_032_dataset = Dataset(label_dir, img_dir_speckle_032, transform=transform)
speckle_064_dataset = Dataset(label_dir, img_dir_speckle_064, transform=transform)
speckle_128_dataset = Dataset(label_dir, img_dir_speckle_128, transform=transform)

speckle_004_loader = DataLoader(speckle_004_dataset, batch_size=training_size, shuffle=True)
speckle_008_loader = DataLoader(speckle_008_dataset, batch_size=training_size, shuffle=True)
speckle_016_loader = DataLoader(speckle_016_dataset, batch_size=training_size, shuffle=True)
speckle_032_loader = DataLoader(speckle_032_dataset, batch_size=training_size, shuffle=True)
speckle_064_loader = DataLoader(speckle_064_dataset, batch_size=training_size, shuffle=True)
speckle_128_loader = DataLoader(speckle_128_dataset, batch_size=training_size, shuffle=True)


snp_002_dataset = Dataset(label_dir, img_dir_snp_002, transform=transform)
snp_004_dataset = Dataset(label_dir, img_dir_snp_004, transform=transform)
snp_008_dataset = Dataset(label_dir, img_dir_snp_008, transform=transform)
snp_016_dataset = Dataset(label_dir, img_dir_snp_016, transform=transform)
snp_032_dataset = Dataset(label_dir, img_dir_snp_032, transform=transform)
snp_064_dataset = Dataset(label_dir, img_dir_snp_064, transform=transform)

snp_002_loader = DataLoader(snp_002_dataset, batch_size=training_size, shuffle=True)
snp_004_loader = DataLoader(snp_004_dataset, batch_size=training_size, shuffle=True)
snp_008_loader = DataLoader(snp_008_dataset, batch_size=training_size, shuffle=True)
snp_016_loader = DataLoader(snp_016_dataset, batch_size=training_size, shuffle=True)
snp_032_loader = DataLoader(snp_032_dataset, batch_size=training_size, shuffle=True)
snp_064_loader = DataLoader(snp_064_dataset, batch_size=training_size, shuffle=True)


with torch.no_grad():
    for i, (images, labels) in enumerate(original_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_original_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_original_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_original_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_original_20k_30k += (predicted == labels_idx).sum().item()

        total += labels.size(0)


    for i, (images, labels) in enumerate(gaussian_002_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_gaussian_002_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_gaussian_002_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_gaussian_002_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_gaussian_002_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(gaussian_004_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_gaussian_004_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_gaussian_004_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_gaussian_004_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_gaussian_004_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(gaussian_008_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_gaussian_008_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_gaussian_008_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_gaussian_008_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_gaussian_008_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(gaussian_016_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_gaussian_016_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_gaussian_016_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_gaussian_016_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_gaussian_016_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(gaussian_032_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_gaussian_032_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_gaussian_032_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_gaussian_032_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_gaussian_032_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(gaussian_064_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_gaussian_064_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_gaussian_064_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_gaussian_064_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_gaussian_064_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(speckle_004_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_speckle_004_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_speckle_004_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_speckle_004_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_speckle_004_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(speckle_008_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_speckle_008_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_speckle_008_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_speckle_008_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_speckle_008_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(speckle_016_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_speckle_016_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_speckle_016_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_speckle_016_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_speckle_016_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(speckle_032_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_speckle_032_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_speckle_032_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_speckle_032_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_speckle_032_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(speckle_064_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_speckle_064_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_speckle_064_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_speckle_064_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_speckle_064_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(speckle_128_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_speckle_128_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_speckle_128_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_speckle_128_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_speckle_128_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(snp_002_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_snp_002_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_snp_002_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_snp_002_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_snp_002_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(snp_004_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_snp_004_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_snp_004_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_snp_004_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_snp_004_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(snp_008_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_snp_008_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_snp_008_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_snp_008_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_snp_008_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(snp_016_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_snp_016_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_snp_016_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_snp_016_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_snp_016_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(snp_032_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_snp_032_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_snp_032_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_snp_032_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_np_032_20k_30k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(snp_064_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_snp_064_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_snp_064_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_snp_064_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_snp_064_20k_30k += (predicted == labels_idx).sum().item()


accuracy_original_50k_0k = (100 * running_accuracy_original_50k_0k / total)
accuracy_original_40k_10k = (100 * running_accuracy_original_40k_10k / total)
accuracy_original_30k_20k = (100 * running_accuracy_original_30k_20k / total)
accuracy_original_20k_30k = (100 * running_accuracy_original_20k_30k / total)


accuracy_gaussian_002_50k_0k = (100 * running_accuracy_gaussian_002_50k_0k / total)
accuracy_gaussian_002_40k_10k = (100 * running_accuracy_gaussian_002_40k_10k / total)
accuracy_gaussian_002_30k_20k = (100 * running_accuracy_gaussian_002_30k_20k / total)
accuracy_gaussian_002_20k_30k = (100 * running_accuracy_gaussian_002_20k_30k / total)

accuracy_gaussian_004_50k_0k = (100 * running_accuracy_gaussian_004_50k_0k / total)
accuracy_gaussian_004_40k_10k = (100 * running_accuracy_gaussian_004_40k_10k / total)
accuracy_gaussian_004_30k_20k = (100 * running_accuracy_gaussian_004_30k_20k / total)
accuracy_gaussian_004_20k_30k = (100 * running_accuracy_gaussian_004_20k_30k / total)

accuracy_gaussian_008_50k_0k = (100 * running_accuracy_gaussian_008_50k_0k / total)
accuracy_gaussian_008_40k_10k = (100 * running_accuracy_gaussian_008_40k_10k / total)
accuracy_gaussian_008_30k_20k = (100 * running_accuracy_gaussian_008_30k_20k / total)
accuracy_gaussian_008_20k_30k = (100 * running_accuracy_gaussian_008_20k_30k / total)

accuracy_gaussian_016_50k_0k = (100 * running_accuracy_gaussian_016_50k_0k / total)
accuracy_gaussian_016_40k_10k = (100 * running_accuracy_gaussian_016_40k_10k / total)
accuracy_gaussian_016_30k_20k = (100 * running_accuracy_gaussian_016_30k_20k / total)
accuracy_gaussian_016_20k_30k = (100 * running_accuracy_gaussian_016_20k_30k / total)

accuracy_gaussian_032_50k_0k = (100 * running_accuracy_gaussian_032_50k_0k / total)
accuracy_gaussian_032_40k_10k = (100 * running_accuracy_gaussian_032_40k_10k / total)
accuracy_gaussian_032_30k_20k = (100 * running_accuracy_gaussian_032_30k_20k / total)
accuracy_gaussian_032_20k_30k = (100 * running_accuracy_gaussian_032_20k_30k / total)

accuracy_gaussian_064_50k_0k = (100 * running_accuracy_gaussian_064_50k_0k / total)
accuracy_gaussian_064_40k_10k = (100 * running_accuracy_gaussian_064_40k_10k / total)
accuracy_gaussian_064_30k_20k = (100 * running_accuracy_gaussian_064_30k_20k / total)
accuracy_gaussian_064_20k_30k = (100 * running_accuracy_gaussian_064_20k_30k / total)


accuracy_speckle_004_50k_0k = (100 * running_accuracy_speckle_004_50k_0k / total)
accuracy_speckle_004_40k_10k = (100 * running_accuracy_speckle_004_40k_10k / total)
accuracy_speckle_004_30k_20k = (100 * running_accuracy_speckle_004_30k_20k / total)
accuracy_speckle_004_20k_30k = (100 * running_accuracy_speckle_004_20k_30k / total)

accuracy_speckle_008_50k_0k = (100 * running_accuracy_speckle_008_50k_0k / total)
accuracy_speckle_008_40k_10k = (100 * running_accuracy_speckle_008_40k_10k / total)
accuracy_speckle_008_30k_20k = (100 * running_accuracy_speckle_008_30k_20k / total)
accuracy_speckle_008_20k_30k = (100 * running_accuracy_speckle_008_20k_30k / total)

accuracy_speckle_016_50k_0k = (100 * running_accuracy_speckle_016_50k_0k / total)
accuracy_speckle_016_40k_10k = (100 * running_accuracy_speckle_016_40k_10k / total)
accuracy_speckle_016_30k_20k = (100 * running_accuracy_speckle_016_30k_20k / total)
accuracy_speckle_016_20k_30k = (100 * running_accuracy_speckle_016_20k_30k / total)

accuracy_speckle_032_50k_0k = (100 * running_accuracy_speckle_032_50k_0k / total)
accuracy_speckle_032_40k_10k = (100 * running_accuracy_speckle_032_40k_10k / total)
accuracy_speckle_032_30k_20k = (100 * running_accuracy_speckle_032_30k_20k / total)
accuracy_speckle_032_20k_30k = (100 * running_accuracy_speckle_032_20k_30k / total)

accuracy_speckle_064_50k_0k = (100 * running_accuracy_speckle_064_50k_0k / total)
accuracy_speckle_064_40k_10k = (100 * running_accuracy_speckle_064_40k_10k / total)
accuracy_speckle_064_30k_20k = (100 * running_accuracy_speckle_064_30k_20k / total)
accuracy_speckle_064_20k_30k = (100 * running_accuracy_speckle_064_20k_30k / total)

accuracy_speckle_128_50k_0k = (100 * running_accuracy_speckle_128_50k_0k / total)
accuracy_speckle_128_40k_10k = (100 * running_accuracy_speckle_128_40k_10k / total)
accuracy_speckle_128_30k_20k = (100 * running_accuracy_speckle_128_30k_20k / total)
accuracy_speckle_128_20k_30k = (100 * running_accuracy_speckle_128_20k_30k / total)


accuracy_snp_002_50k_0k = (100 * running_accuracy_snp_002_50k_0k / total)
accuracy_snp_002_40k_10k = (100 * running_accuracy_snp_002_40k_10k / total)
accuracy_snp_002_30k_20k = (100 * running_accuracy_snp_002_30k_20k / total)
accuracy_snp_002_20k_30k = (100 * running_accuracy_snp_002_20k_30k / total)

accuracy_snp_004_50k_0k = (100 * running_accuracy_snp_004_50k_0k / total)
accuracy_snp_004_40k_10k = (100 * running_accuracy_snp_004_40k_10k / total)
accuracy_snp_004_30k_20k = (100 * running_accuracy_snp_004_30k_20k / total)
accuracy_snp_004_20k_30k = (100 * running_accuracy_snp_004_20k_30k / total)

accuracy_snp_008_50k_0k = (100 * running_accuracy_snp_008_50k_0k / total)
accuracy_snp_008_40k_10k = (100 * running_accuracy_snp_008_40k_10k / total)
accuracy_snp_008_30k_20k = (100 * running_accuracy_snp_008_30k_20k / total)
accuracy_snp_008_20k_30k = (100 * running_accuracy_snp_008_20k_30k / total)

accuracy_snp_016_50k_0k = (100 * running_accuracy_snp_016_50k_0k / total)
accuracy_snp_016_40k_10k = (100 * running_accuracy_snp_016_40k_10k / total)
accuracy_snp_016_30k_20k = (100 * running_accuracy_snp_016_30k_20k / total)
accuracy_snp_016_20k_30k = (100 * running_accuracy_snp_016_20k_30k / total)

accuracy_snp_032_50k_0k = (100 * running_accuracy_snp_032_50k_0k / total)
accuracy_snp_032_40k_10k = (100 * running_accuracy_snp_032_40k_10k / total)
accuracy_snp_032_30k_20k = (100 * running_accuracy_snp_032_30k_20k / total)
accuracy_snp_032_20k_30k = (100 * running_accuracy_snp_032_20k_30k / total)

accuracy_snp_064_50k_0k = (100 * running_accuracy_snp_064_50k_0k / total)
accuracy_snp_064_40k_10k = (100 * running_accuracy_snp_064_40k_10k / total)
accuracy_snp_064_30k_20k = (100 * running_accuracy_snp_064_30k_20k / total)
accuracy_snp_064_20k_30k = (100 * running_accuracy_snp_064_20k_30k / total)


stats = []

stats.append('original_50k_0k: {:.2f}%'.format(accuracy_original_50k_0k))
stats.append('original_40k_10k: {:.2f}%'.format(accuracy_original_40k_10k))
stats.append('original_30k_20k: {:.2f}%'.format(accuracy_original_30k_20k))
stats.append('original_20k_30k: {:.2f}%'.format(accuracy_original_20k_30k))
stats.append('')
stats.append('')
stats.append('gaussian_002_50k_0k: {:.2f}%'.format(accuracy_gaussian_002_50k_0k))
stats.append('gaussian_002_40k_10k: {:.2f}%'.format(accuracy_gaussian_002_40k_10k))
stats.append('gaussian_002_30k_20k: {:.2f}%'.format(accuracy_gaussian_002_30k_20k))
stats.append('gaussian_002_20k_30k: {:.2f}%'.format(accuracy_gaussian_002_20k_30k))
stats.append('')
stats.append('gaussian_004_50k_0k: {:.2f}%'.format(accuracy_gaussian_004_50k_0k))
stats.append('gaussian_004_40k_10k: {:.2f}%'.format(accuracy_gaussian_004_40k_10k))
stats.append('gaussian_004_30k_20k: {:.2f}%'.format(accuracy_gaussian_004_30k_20k))
stats.append('gaussian_004_20k_30k: {:.2f}%'.format(accuracy_gaussian_004_20k_30k))
stats.append('')
stats.append('gaussian_008_50k_0k: {:.2f}%'.format(accuracy_gaussian_008_50k_0k))
stats.append('gaussian_008_40k_10k: {:.2f}%'.format(accuracy_gaussian_008_40k_10k))
stats.append('gaussian_008_30k_20k: {:.2f}%'.format(accuracy_gaussian_008_30k_20k))
stats.append('gaussian_008_20k_30k: {:.2f}%'.format(accuracy_gaussian_008_20k_30k))
stats.append('')
stats.append('gaussian_016_50k_0k: {:.2f}%'.format(accuracy_gaussian_016_50k_0k))
stats.append('gaussian_016_40k_10k: {:.2f}%'.format(accuracy_gaussian_016_40k_10k))
stats.append('gaussian_016_30k_20k: {:.2f}%'.format(accuracy_gaussian_016_30k_20k))
stats.append('gaussian_016_20k_30k: {:.2f}%'.format(accuracy_gaussian_016_20k_30k))
stats.append('')
stats.append('gaussian_032_50k_0k: {:.2f}%'.format(accuracy_gaussian_032_50k_0k))
stats.append('gaussian_032_40k_10k: {:.2f}%'.format(accuracy_gaussian_032_40k_10k))
stats.append('gaussian_032_30k_20k: {:.2f}%'.format(accuracy_gaussian_032_30k_20k))
stats.append('gaussian_032_20k_30k: {:.2f}%'.format(accuracy_gaussian_032_20k_30k))
stats.append('')
stats.append('gaussian_064_50k_0k: {:.2f}%'.format(accuracy_gaussian_064_50k_0k))
stats.append('gaussian_064_40k_10k: {:.2f}%'.format(accuracy_gaussian_064_40k_10k))
stats.append('gaussian_064_30k_20k: {:.2f}%'.format(accuracy_gaussian_064_30k_20k))
stats.append('gaussian_064_20k_30k: {:.2f}%'.format(accuracy_gaussian_064_20k_30k))
stats.append('')
stats.append('')
stats.append('speckle_004_50k_0k: {:.2f}%'.format(accuracy_speckle_004_50k_0k))
stats.append('speckle_004_40k_10k: {:.2f}%'.format(accuracy_speckle_004_40k_10k))
stats.append('speckle_004_30k_20k: {:.2f}%'.format(accuracy_speckle_004_30k_20k))
stats.append('speckle_004_20k_30k: {:.2f}%'.format(accuracy_speckle_004_20k_30k))
stats.append('')
stats.append('speckle_008_50k_0k: {:.2f}%'.format(accuracy_speckle_008_50k_0k))
stats.append('speckle_008_40k_10k: {:.2f}%'.format(accuracy_speckle_008_40k_10k))
stats.append('speckle_008_30k_20k: {:.2f}%'.format(accuracy_speckle_008_30k_20k))
stats.append('speckle_008_20k_30k: {:.2f}%'.format(accuracy_speckle_008_20k_30k))
stats.append('')
stats.append('speckle_016_50k_0k: {:.2f}%'.format(accuracy_speckle_016_50k_0k))
stats.append('speckle_016_40k_10k: {:.2f}%'.format(accuracy_speckle_016_40k_10k))
stats.append('speckle_016_30k_20k: {:.2f}%'.format(accuracy_speckle_016_30k_20k))
stats.append('speckle_016_20k_30k: {:.2f}%'.format(accuracy_speckle_016_20k_30k))
stats.append('')
stats.append('speckle_032_50k_0k: {:.2f}%'.format(accuracy_speckle_032_50k_0k))
stats.append('speckle_032_40k_10k: {:.2f}%'.format(accuracy_speckle_032_40k_10k))
stats.append('speckle_032_30k_20k: {:.2f}%'.format(accuracy_speckle_032_30k_20k))
stats.append('speckle_032_20k_30k: {:.2f}%'.format(accuracy_speckle_032_20k_30k))
stats.append('')
stats.append('speckle_064_50k_0k: {:.2f}%'.format(accuracy_speckle_064_50k_0k))
stats.append('speckle_064_40k_10k: {:.2f}%'.format(accuracy_speckle_064_40k_10k))
stats.append('speckle_064_30k_20k: {:.2f}%'.format(accuracy_speckle_064_30k_20k))
stats.append('speckle_064_20k_30k: {:.2f}%'.format(accuracy_speckle_064_20k_30k))
stats.append('')
stats.append('speckle_128_50k_0k: {:.2f}%'.format(accuracy_speckle_128_50k_0k))
stats.append('speckle_128_40k_10k: {:.2f}%'.format(accuracy_speckle_128_40k_10k))
stats.append('speckle_128_30k_20k: {:.2f}%'.format(accuracy_speckle_128_30k_20k))
stats.append('speckle_128_20k_30k: {:.2f}%'.format(accuracy_speckle_128_20k_30k))
stats.append('')
stats.append('')
stats.append('snp_002_50k_0k: {:.2f}%'.format(accuracy_snp_002_50k_0k))
stats.append('snp_002_40k_10k: {:.2f}%'.format(accuracy_snp_002_40k_10k))
stats.append('snp_002_30k_20k: {:.2f}%'.format(accuracy_snp_002_30k_20k))
stats.append('snp_002_20k_30k: {:.2f}%'.format(accuracy_snp_002_20k_30k))
stats.append('')
stats.append('snp_004_50k_0k: {:.2f}%'.format(accuracy_snp_004_50k_0k))
stats.append('snp_004_40k_10k: {:.2f}%'.format(accuracy_snp_004_40k_10k))
stats.append('snp_004_30k_20k: {:.2f}%'.format(accuracy_snp_004_30k_20k))
stats.append('snp_004_20k_30k: {:.2f}%'.format(accuracy_snp_004_20k_30k))
stats.append('')
stats.append('snp_008_50k_0k: {:.2f}%'.format(accuracy_snp_008_50k_0k))
stats.append('snp_008_40k_10k: {:.2f}%'.format(accuracy_snp_008_40k_10k))
stats.append('snp_008_30k_20k: {:.2f}%'.format(accuracy_snp_008_30k_20k))
stats.append('snp_008_20k_30k: {:.2f}%'.format(accuracy_snp_008_20k_30k))
stats.append('')
stats.append('snp_016_50k_0k: {:.2f}%'.format(accuracy_snp_016_50k_0k))
stats.append('snp_016_40k_10k: {:.2f}%'.format(accuracy_snp_016_40k_10k))
stats.append('snp_016_30k_20k: {:.2f}%'.format(accuracy_snp_016_30k_20k))
stats.append('snp_016_20k_30k: {:.2f}%'.format(accuracy_snp_016_20k_30k))
stats.append('')
stats.append('snp_032_50k_0k: {:.2f}%'.format(accuracy_snp_032_50k_0k))
stats.append('snp_032_40k_10k: {:.2f}%'.format(accuracy_snp_032_40k_10k))
stats.append('snp_032_30k_20k: {:.2f}%'.format(accuracy_snp_032_30k_20k))
stats.append('snp_032_20k_30k: {:.2f}%'.format(accuracy_snp_032_20k_30k))
stats.append('')
stats.append('snp_064_50k_0k: {:.2f}%'.format(accuracy_snp_064_50k_0k))
stats.append('snp_064_40k_10k: {:.2f}%'.format(accuracy_snp_064_40k_10k))
stats.append('snp_064_30k_20k: {:.2f}%'.format(accuracy_snp_064_30k_20k))
stats.append('snp_064_20k_30k: {:.2f}%'.format(accuracy_snp_064_20k_30k))


with open(r'results.txt', 'w') as fp:
    for parameter in stats:
        fp.write('{}\n'.format(parameter))



# create plot
