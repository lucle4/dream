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

img_dir_noise_1_1 = os.path.join(directory, 'evaluation/blur (1)')
img_dir_noise_1_2 = os.path.join(directory, 'evaluation/blur (2)')
img_dir_noise_1_3 = os.path.join(directory, 'evaluation/blur (3)')
img_dir_noise_1_4 = os.path.join(directory, 'evaluation/blur (5)')
img_dir_noise_1_5 = os.path.join(directory, 'evaluation/blur (9)')
img_dir_noise_1_6 = os.path.join(directory, 'evaluation/blur (11)')

img_dir_noise_2_1 = os.path.join(directory, 'evaluation/contrast (0.8)')
img_dir_noise_2_2 = os.path.join(directory, 'evaluation/contrast (0.6)')
img_dir_noise_2_3 = os.path.join(directory, 'evaluation/contrast (0.4)')
img_dir_noise_2_4 = os.path.join(directory, 'evaluation/contrast (0.3)')
img_dir_noise_2_5 = os.path.join(directory, 'evaluation/contrast (0.05)')
img_dir_noise_2_6 = os.path.join(directory, 'evaluation/contrast (0.02)')

img_dir_noise_3_1 = os.path.join(directory, 'evaluation/highpass (1)')
img_dir_noise_3_2 = os.path.join(directory, 'evaluation/highpass (4)')
img_dir_noise_3_3 = os.path.join(directory, 'evaluation/highpass (6)')
img_dir_noise_3_4 = os.path.join(directory, 'evaluation/highpass (12)')
img_dir_noise_3_5 = os.path.join(directory, 'evaluation/highpass (15)')
img_dir_noise_3_6 = os.path.join(directory, 'evaluation/highpass (20)')


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

model_10k_40k = resnet50(weights=None)
model_10k_40k.fc = nn.Linear(2048, 10)

model_0k_50k = resnet50(weights=None)
model_0k_50k.fc = nn.Linear(2048, 10)


if torch.cuda.is_available():
    model_50k_0k.cuda()
    model_40k_10k.cuda()
    model_30k_20k.cuda()
    model_20k_30k.cuda()
    model_10k_40k.cuda()
    model_0k_50k.cuda()


model_50k_0k.load_state_dict(torch.load(os.path.join(directory, 'checkpoints_50k_0k/checkpoint epoch 100.pt')))
model_50k_0k.eval()

model_40k_10k.load_state_dict(torch.load(os.path.join(directory, 'checkpoints_40k_10k/checkpoint epoch 100.pt')))
model_40k_10k.eval()

model_30k_20k.load_state_dict(torch.load(os.path.join(directory, 'checkpoints_30k_20k/checkpoint epoch 100.pt')))
model_30k_20k.eval()

model_20k_30k.load_state_dict(torch.load(os.path.join(directory, 'checkpoints_20k_30k/checkpoint epoch 100.pt')))
model_20k_30k.eval()

model_10k_40k.load_state_dict(torch.load(os.path.join(directory, 'checkpoints_10k_40k/checkpoint epoch 100.pt')))
model_10k_40k.eval()

model_0k_50k.load_state_dict(torch.load(os.path.join(directory, 'checkpoints_0k_50k/checkpoint epoch 100.pt')))
model_0k_50k.eval()



running_accuracy_original_50k_0k = 0.0
running_accuracy_original_40k_10k = 0.0
running_accuracy_original_30k_20k = 0.0
running_accuracy_original_20k_30k = 0.0
running_accuracy_original_10k_40k = 0.0
running_accuracy_original_0k_50k = 0.0


running_accuracy_noise_1_1_50k_0k = 0.0
running_accuracy_noise_1_1_40k_10k = 0.0
running_accuracy_noise_1_1_30k_20k = 0.0
running_accuracy_noise_1_1_20k_30k = 0.0
running_accuracy_noise_1_1_10k_40k = 0.0
running_accuracy_noise_1_1_0k_50k = 0.0

running_accuracy_noise_1_2_50k_0k = 0.0
running_accuracy_noise_1_2_40k_10k = 0.0
running_accuracy_noise_1_2_30k_20k = 0.0
running_accuracy_noise_1_2_20k_30k = 0.0
running_accuracy_noise_1_2_10k_40k = 0.0
running_accuracy_noise_1_2_0k_50k = 0.0

running_accuracy_noise_1_3_50k_0k = 0.0
running_accuracy_noise_1_3_40k_10k = 0.0
running_accuracy_noise_1_3_30k_20k = 0.0
running_accuracy_noise_1_3_20k_30k = 0.0
running_accuracy_noise_1_3_10k_40k = 0.0
running_accuracy_noise_1_3_0k_50k = 0.0

running_accuracy_noise_1_4_50k_0k = 0.0
running_accuracy_noise_1_4_40k_10k = 0.0
running_accuracy_noise_1_4_30k_20k = 0.0
running_accuracy_noise_1_4_20k_30k = 0.0
running_accuracy_noise_1_4_10k_40k = 0.0
running_accuracy_noise_1_4_0k_50k = 0.0

running_accuracy_noise_1_5_50k_0k = 0.0
running_accuracy_noise_1_5_40k_10k = 0.0
running_accuracy_noise_1_5_30k_20k = 0.0
running_accuracy_noise_1_5_20k_30k = 0.0
running_accuracy_noise_1_5_10k_40k = 0.0
running_accuracy_noise_1_5_0k_50k = 0.0

running_accuracy_noise_1_6_50k_0k = 0.0
running_accuracy_noise_1_6_40k_10k = 0.0
running_accuracy_noise_1_6_30k_20k = 0.0
running_accuracy_noise_1_6_20k_30k = 0.0
running_accuracy_noise_1_6_10k_40k = 0.0
running_accuracy_noise_1_6_0k_50k = 0.0


running_accuracy_noise_2_1_50k_0k = 0.0
running_accuracy_noise_2_1_40k_10k = 0.0
running_accuracy_noise_2_1_30k_20k = 0.0
running_accuracy_noise_2_1_20k_30k = 0.0
running_accuracy_noise_2_1_10k_40k = 0.0
running_accuracy_noise_2_1_0k_50k = 0.0

running_accuracy_noise_2_2_50k_0k = 0.0
running_accuracy_noise_2_2_40k_10k = 0.0
running_accuracy_noise_2_2_30k_20k = 0.0
running_accuracy_noise_2_2_20k_30k = 0.0
running_accuracy_noise_2_2_10k_40k = 0.0
running_accuracy_noise_2_2_0k_50k = 0.0

running_accuracy_noise_2_3_50k_0k = 0.0
running_accuracy_noise_2_3_40k_10k = 0.0
running_accuracy_noise_2_3_30k_20k = 0.0
running_accuracy_noise_2_3_20k_30k = 0.0
running_accuracy_noise_2_3_10k_40k = 0.0
running_accuracy_noise_2_3_0k_50k = 0.0

running_accuracy_noise_2_4_50k_0k = 0.0
running_accuracy_noise_2_4_40k_10k = 0.0
running_accuracy_noise_2_4_30k_20k = 0.0
running_accuracy_noise_2_4_20k_30k = 0.0
running_accuracy_noise_2_4_10k_40k = 0.0
running_accuracy_noise_2_4_0k_50k = 0.0

running_accuracy_noise_2_5_50k_0k = 0.0
running_accuracy_noise_2_5_40k_10k = 0.0
running_accuracy_noise_2_5_30k_20k = 0.0
running_accuracy_noise_2_5_20k_30k = 0.0
running_accuracy_noise_2_5_10k_40k = 0.0
running_accuracy_noise_2_5_0k_50k = 0.0

running_accuracy_noise_2_6_50k_0k = 0.0
running_accuracy_noise_2_6_40k_10k = 0.0
running_accuracy_noise_2_6_30k_20k = 0.0
running_accuracy_noise_2_6_20k_30k = 0.0
running_accuracy_noise_2_6_10k_40k = 0.0
running_accuracy_noise_2_6_0k_50k = 0.0


running_accuracy_noise_3_1_50k_0k = 0.0
running_accuracy_noise_3_1_40k_10k = 0.0
running_accuracy_noise_3_1_30k_20k = 0.0
running_accuracy_noise_3_1_20k_30k = 0.0
running_accuracy_noise_3_1_10k_40k = 0.0
running_accuracy_noise_3_1_0k_50k = 0.0

running_accuracy_noise_3_2_50k_0k = 0.0
running_accuracy_noise_3_2_40k_10k = 0.0
running_accuracy_noise_3_2_30k_20k = 0.0
running_accuracy_noise_3_2_20k_30k = 0.0
running_accuracy_noise_3_2_10k_40k = 0.0
running_accuracy_noise_3_2_0k_50k = 0.0

running_accuracy_noise_3_3_50k_0k = 0.0
running_accuracy_noise_3_3_40k_10k = 0.0
running_accuracy_noise_3_3_30k_20k = 0.0
running_accuracy_noise_3_3_20k_30k = 0.0
running_accuracy_noise_3_3_10k_40k = 0.0
running_accuracy_noise_3_3_0k_50k = 0.0

running_accuracy_noise_3_4_50k_0k = 0.0
running_accuracy_noise_3_4_40k_10k = 0.0
running_accuracy_noise_3_4_30k_20k = 0.0
running_accuracy_noise_3_4_20k_30k = 0.0
running_accuracy_noise_3_4_10k_40k = 0.0
running_accuracy_noise_3_4_0k_50k = 0.0

running_accuracy_noise_3_5_50k_0k = 0.0
running_accuracy_noise_3_5_40k_10k = 0.0
running_accuracy_noise_3_5_30k_20k = 0.0
running_accuracy_noise_3_5_20k_30k = 0.0
running_accuracy_noise_3_5_10k_40k = 0.0
running_accuracy_noise_3_5_0k_50k = 0.0

running_accuracy_noise_3_6_50k_0k = 0.0
running_accuracy_noise_3_6_40k_10k = 0.0
running_accuracy_noise_3_6_30k_20k = 0.0
running_accuracy_noise_3_6_20k_30k = 0.0
running_accuracy_noise_3_6_10k_40k = 0.0
running_accuracy_noise_3_6_0k_50k = 0.0


total = 0


original_dataset = Dataset(label_dir, img_dir_original, transform=transform)
training_size = len(original_dataset)

original_loader = DataLoader(original_dataset, batch_size=training_size, shuffle=True)


noise_1_1_dataset = Dataset(label_dir, img_dir_noise_1_1, transform=transform)
noise_1_2_dataset = Dataset(label_dir, img_dir_noise_1_2, transform=transform)
noise_1_3_dataset = Dataset(label_dir, img_dir_noise_1_3, transform=transform)
noise_1_4_dataset = Dataset(label_dir, img_dir_noise_1_4, transform=transform)
noise_1_5_dataset = Dataset(label_dir, img_dir_noise_1_5, transform=transform)
noise_1_6_dataset = Dataset(label_dir, img_dir_noise_1_6, transform=transform)

noise_1_1_loader = DataLoader(noise_1_1_dataset, batch_size=training_size, shuffle=True)
noise_1_2_loader = DataLoader(noise_1_2_dataset, batch_size=training_size, shuffle=True)
noise_1_3_loader = DataLoader(noise_1_3_dataset, batch_size=training_size, shuffle=True)
noise_1_4_loader = DataLoader(noise_1_4_dataset, batch_size=training_size, shuffle=True)
noise_1_5_loader = DataLoader(noise_1_5_dataset, batch_size=training_size, shuffle=True)
noise_1_6_loader = DataLoader(noise_1_6_dataset, batch_size=training_size, shuffle=True)


noise_2_1_dataset = Dataset(label_dir, img_dir_noise_2_1, transform=transform)
noise_2_2_dataset = Dataset(label_dir, img_dir_noise_2_2, transform=transform)
noise_2_3_dataset = Dataset(label_dir, img_dir_noise_2_3, transform=transform)
noise_2_4_dataset = Dataset(label_dir, img_dir_noise_2_4, transform=transform)
noise_2_5_dataset = Dataset(label_dir, img_dir_noise_2_5, transform=transform)
noise_2_6_dataset = Dataset(label_dir, img_dir_noise_2_6, transform=transform)

noise_2_1_loader = DataLoader(noise_2_1_dataset, batch_size=training_size, shuffle=True)
noise_2_2_loader = DataLoader(noise_2_2_dataset, batch_size=training_size, shuffle=True)
noise_2_3_loader = DataLoader(noise_2_3_dataset, batch_size=training_size, shuffle=True)
noise_2_4_loader = DataLoader(noise_2_4_dataset, batch_size=training_size, shuffle=True)
noise_2_5_loader = DataLoader(noise_2_5_dataset, batch_size=training_size, shuffle=True)
noise_2_6_loader = DataLoader(noise_2_6_dataset, batch_size=training_size, shuffle=True)


noise_3_1_dataset = Dataset(label_dir, img_dir_noise_3_1, transform=transform)
noise_3_2_dataset = Dataset(label_dir, img_dir_noise_3_2, transform=transform)
noise_3_3_dataset = Dataset(label_dir, img_dir_noise_3_3, transform=transform)
noise_3_4_dataset = Dataset(label_dir, img_dir_noise_3_4, transform=transform)
noise_3_5_dataset = Dataset(label_dir, img_dir_noise_3_5, transform=transform)
noise_3_6_dataset = Dataset(label_dir, img_dir_noise_3_6, transform=transform)

noise_3_1_loader = DataLoader(noise_3_1_dataset, batch_size=training_size, shuffle=True)
noise_3_2_loader = DataLoader(noise_3_2_dataset, batch_size=training_size, shuffle=True)
noise_3_3_loader = DataLoader(noise_3_3_dataset, batch_size=training_size, shuffle=True)
noise_3_4_loader = DataLoader(noise_3_4_dataset, batch_size=training_size, shuffle=True)
noise_3_5_loader = DataLoader(noise_3_5_dataset, batch_size=training_size, shuffle=True)
noise_3_6_loader = DataLoader(noise_3_6_dataset, batch_size=training_size, shuffle=True)


with torch.no_grad():
    for i, (images, labels) in enumerate(original_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_original_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_original_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_original_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_original_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_original_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_original_0k_50k += (predicted == labels_idx).sum().item()

        total += labels.size(0)


    for i, (images, labels) in enumerate(noise_1_1_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_1_1_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_1_1_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_1_1_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_1_1_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_1_1_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_1_1_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_1_2_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_1_2_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_1_2_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_1_2_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_1_2_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_1_2_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_1_2_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_1_3_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_1_3_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_1_3_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_1_3_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_1_3_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_1_3_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_1_3_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_1_4_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_1_4_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_1_4_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_1_4_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_1_4_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_1_4_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_1_4_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_1_5_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_1_5_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_1_5_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_1_5_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_1_5_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_1_5_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_1_5_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_1_6_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_1_6_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_1_6_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_1_6_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_1_6_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_1_6_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_1_6_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_2_1_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_2_1_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_2_1_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_2_1_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_2_1_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_2_1_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_2_1_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_2_2_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_2_2_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_2_2_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_2_2_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_2_2_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_2_2_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_2_2_0k_50k += (predicted == labels_idx).sum().item()



    for i, (images, labels) in enumerate(noise_2_3_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_2_3_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_2_3_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_2_3_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_2_3_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_2_3_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_2_3_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_2_4_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_2_4_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_2_4_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_2_4_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_2_4_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_2_4_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_2_4_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_2_5_loader):
          images = images.to(device)
          labels = labels.to(device)
          labels_idx = torch.argmax(labels, 1)

          output_50k_0k = model_50k_0k(images.float())
          output_40k_10k = model_40k_10k(images.float())
          output_30k_20k = model_30k_20k(images.float())
          output_20k_30k = model_20k_30k(images.float())
          output_10k_40k = model_10k_40k(images.float())
          output_0k_50k = model_0k_50k(images.float())

          _, predicted = torch.max(output_50k_0k, 1)
          running_accuracy_noise_2_5_50k_0k += (predicted == labels_idx).sum().item()

          _, predicted = torch.max(output_40k_10k, 1)
          running_accuracy_noise_2_5_40k_10k += (predicted == labels_idx).sum().item()

          _, predicted = torch.max(output_30k_20k, 1)
          running_accuracy_noise_2_5_30k_20k += (predicted == labels_idx).sum().item()

          _, predicted = torch.max(output_20k_30k, 1)
          running_accuracy_noise_2_5_20k_30k += (predicted == labels_idx).sum().item()

          _, predicted = torch.max(output_10k_40k, 1)
          running_accuracy_noise_2_5_10k_40k += (predicted == labels_idx).sum().item()

          _, predicted = torch.max(output_0k_50k, 1)
          running_accuracy_noise_2_5_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_2_6_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_2_6_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_2_6_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_2_6_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_2_6_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_2_6_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_2_6_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_3_1_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_3_1_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_3_1_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_3_1_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_3_1_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_3_1_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_3_1_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_3_2_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_3_2_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_3_2_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_3_2_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_3_2_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_3_2_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_3_2_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_3_3_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_3_3_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_3_3_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_3_3_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_3_3_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_3_3_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_3_3_0k_50k += (predicted == labels_idx).sum().item()



    for i, (images, labels) in enumerate(noise_3_4_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_3_4_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_3_4_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_3_4_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_3_4_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_3_4_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_3_4_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_3_5_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_3_5_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_3_5_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_3_5_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_3_5_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_3_5_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_3_5_0k_50k += (predicted == labels_idx).sum().item()


    for i, (images, labels) in enumerate(noise_3_6_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels_idx = torch.argmax(labels, 1)

        output_50k_0k = model_50k_0k(images.float())
        output_40k_10k = model_40k_10k(images.float())
        output_30k_20k = model_30k_20k(images.float())
        output_20k_30k = model_20k_30k(images.float())
        output_10k_40k = model_10k_40k(images.float())
        output_0k_50k = model_0k_50k(images.float())

        _, predicted = torch.max(output_50k_0k, 1)
        running_accuracy_noise_3_6_50k_0k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_40k_10k, 1)
        running_accuracy_noise_3_6_40k_10k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_30k_20k, 1)
        running_accuracy_noise_3_6_30k_20k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_20k_30k, 1)
        running_accuracy_noise_3_6_20k_30k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_10k_40k, 1)
        running_accuracy_noise_3_6_10k_40k += (predicted == labels_idx).sum().item()

        _, predicted = torch.max(output_0k_50k, 1)
        running_accuracy_noise_3_6_0k_50k += (predicted == labels_idx).sum().item()


accuracy_original_50k_0k = (running_accuracy_original_50k_0k / total)
accuracy_original_40k_10k = (running_accuracy_original_40k_10k / total)
accuracy_original_30k_20k = (running_accuracy_original_30k_20k / total)
accuracy_original_20k_30k = (running_accuracy_original_20k_30k / total)
accuracy_original_10k_40k = (running_accuracy_original_10k_40k / total)
accuracy_original_0k_50k = (running_accuracy_original_0k_50k / total)


accuracy_noise_1_1_50k_0k = (running_accuracy_noise_1_1_50k_0k / total)
accuracy_noise_1_1_40k_10k = (running_accuracy_noise_1_1_40k_10k / total)
accuracy_noise_1_1_30k_20k = (running_accuracy_noise_1_1_30k_20k / total)
accuracy_noise_1_1_20k_30k = (running_accuracy_noise_1_1_20k_30k / total)
accuracy_noise_1_1_10k_40k = (running_accuracy_noise_1_1_10k_40k / total)
accuracy_noise_1_1_0k_50k = (running_accuracy_noise_1_1_0k_50k / total)

accuracy_noise_1_2_50k_0k = (running_accuracy_noise_1_2_50k_0k / total)
accuracy_noise_1_2_40k_10k = (running_accuracy_noise_1_2_40k_10k / total)
accuracy_noise_1_2_30k_20k = (running_accuracy_noise_1_2_30k_20k / total)
accuracy_noise_1_2_20k_30k = (running_accuracy_noise_1_2_20k_30k / total)
accuracy_noise_1_2_10k_40k = (running_accuracy_noise_1_2_10k_40k / total)
accuracy_noise_1_2_0k_50k = (running_accuracy_noise_1_2_0k_50k / total)

accuracy_noise_1_3_50k_0k = (running_accuracy_noise_1_3_50k_0k / total)
accuracy_noise_1_3_40k_10k = (running_accuracy_noise_1_3_40k_10k / total)
accuracy_noise_1_3_30k_20k = (running_accuracy_noise_1_3_30k_20k / total)
accuracy_noise_1_3_20k_30k = (running_accuracy_noise_1_3_20k_30k / total)
accuracy_noise_1_3_10k_40k = (running_accuracy_noise_1_3_10k_40k / total)
accuracy_noise_1_3_0k_50k = (running_accuracy_noise_1_3_0k_50k / total)

accuracy_noise_1_4_50k_0k = (running_accuracy_noise_1_4_50k_0k / total)
accuracy_noise_1_4_40k_10k = (running_accuracy_noise_1_4_40k_10k / total)
accuracy_noise_1_4_30k_20k = (running_accuracy_noise_1_4_30k_20k / total)
accuracy_noise_1_4_20k_30k = (running_accuracy_noise_1_4_20k_30k / total)
accuracy_noise_1_4_10k_40k = (running_accuracy_noise_1_4_10k_40k / total)
accuracy_noise_1_4_0k_50k = (running_accuracy_noise_1_4_0k_50k / total)

accuracy_noise_1_5_50k_0k = (running_accuracy_noise_1_5_50k_0k / total)
accuracy_noise_1_5_40k_10k = (running_accuracy_noise_1_5_40k_10k / total)
accuracy_noise_1_5_30k_20k = (running_accuracy_noise_1_5_30k_20k / total)
accuracy_noise_1_5_20k_30k = (running_accuracy_noise_1_5_20k_30k / total)
accuracy_noise_1_5_10k_40k = (running_accuracy_noise_1_5_10k_40k / total)
accuracy_noise_1_5_0k_50k = (running_accuracy_noise_1_5_0k_50k / total)

accuracy_noise_1_6_50k_0k = (running_accuracy_noise_1_6_50k_0k / total)
accuracy_noise_1_6_40k_10k = (running_accuracy_noise_1_6_40k_10k / total)
accuracy_noise_1_6_30k_20k = (running_accuracy_noise_1_6_30k_20k / total)
accuracy_noise_1_6_20k_30k = (running_accuracy_noise_1_6_20k_30k / total)
accuracy_noise_1_6_10k_40k = (running_accuracy_noise_1_6_10k_40k / total)
accuracy_noise_1_6_0k_50k = (running_accuracy_noise_1_6_0k_50k / total)


accuracy_noise_2_1_50k_0k = (running_accuracy_noise_2_1_50k_0k / total)
accuracy_noise_2_1_40k_10k = (running_accuracy_noise_2_1_40k_10k / total)
accuracy_noise_2_1_30k_20k = (running_accuracy_noise_2_1_30k_20k / total)
accuracy_noise_2_1_20k_30k = (running_accuracy_noise_2_1_20k_30k / total)
accuracy_noise_2_1_10k_40k = (running_accuracy_noise_2_1_10k_40k / total)
accuracy_noise_2_1_0k_50k = (running_accuracy_noise_2_1_0k_50k / total)

accuracy_noise_2_2_50k_0k = (running_accuracy_noise_2_2_50k_0k / total)
accuracy_noise_2_2_40k_10k = (running_accuracy_noise_2_2_40k_10k / total)
accuracy_noise_2_2_30k_20k = (running_accuracy_noise_2_2_30k_20k / total)
accuracy_noise_2_2_20k_30k = (running_accuracy_noise_2_2_20k_30k / total)
accuracy_noise_2_2_10k_40k = (running_accuracy_noise_2_2_10k_40k / total)
accuracy_noise_2_2_0k_50k = (running_accuracy_noise_2_2_0k_50k / total)

accuracy_noise_2_3_50k_0k = (running_accuracy_noise_2_3_50k_0k / total)
accuracy_noise_2_3_40k_10k = (running_accuracy_noise_2_3_40k_10k / total)
accuracy_noise_2_3_30k_20k = (running_accuracy_noise_2_3_30k_20k / total)
accuracy_noise_2_3_20k_30k = (running_accuracy_noise_2_3_20k_30k / total)
accuracy_noise_2_3_10k_40k = (running_accuracy_noise_2_3_10k_40k / total)
accuracy_noise_2_3_0k_50k = (running_accuracy_noise_2_3_0k_50k / total)

accuracy_noise_2_4_50k_0k = (running_accuracy_noise_2_4_50k_0k / total)
accuracy_noise_2_4_40k_10k = (running_accuracy_noise_2_4_40k_10k / total)
accuracy_noise_2_4_30k_20k = (running_accuracy_noise_2_4_30k_20k / total)
accuracy_noise_2_4_20k_30k = (running_accuracy_noise_2_4_20k_30k / total)
accuracy_noise_2_4_10k_40k = (running_accuracy_noise_2_4_10k_40k / total)
accuracy_noise_2_4_0k_50k = (running_accuracy_noise_2_4_0k_50k / total)

accuracy_noise_2_5_50k_0k = (running_accuracy_noise_2_5_50k_0k / total)
accuracy_noise_2_5_40k_10k = (running_accuracy_noise_2_5_40k_10k / total)
accuracy_noise_2_5_30k_20k = (running_accuracy_noise_2_5_30k_20k / total)
accuracy_noise_2_5_20k_30k = (running_accuracy_noise_2_5_20k_30k / total)
accuracy_noise_2_5_10k_40k = (running_accuracy_noise_2_5_10k_40k / total)
accuracy_noise_2_5_0k_50k = (running_accuracy_noise_2_5_0k_50k / total)

accuracy_noise_2_6_50k_0k = (running_accuracy_noise_2_6_50k_0k / total)
accuracy_noise_2_6_40k_10k = (running_accuracy_noise_2_6_40k_10k / total)
accuracy_noise_2_6_30k_20k = (running_accuracy_noise_2_6_30k_20k / total)
accuracy_noise_2_6_20k_30k = (running_accuracy_noise_2_6_20k_30k / total)
accuracy_noise_2_6_10k_40k = (running_accuracy_noise_2_6_10k_40k / total)
accuracy_noise_2_6_0k_50k = (running_accuracy_noise_2_6_0k_50k / total)


accuracy_noise_3_1_50k_0k = (running_accuracy_noise_3_1_50k_0k / total)
accuracy_noise_3_1_40k_10k = (running_accuracy_noise_3_1_40k_10k / total)
accuracy_noise_3_1_30k_20k = (running_accuracy_noise_3_1_30k_20k / total)
accuracy_noise_3_1_20k_30k = (running_accuracy_noise_3_1_20k_30k / total)
accuracy_noise_3_1_10k_40k = (running_accuracy_noise_3_1_10k_40k / total)
accuracy_noise_3_1_0k_50k = (running_accuracy_noise_3_1_0k_50k / total)

accuracy_noise_3_2_50k_0k = (running_accuracy_noise_3_2_50k_0k / total)
accuracy_noise_3_2_40k_10k = (running_accuracy_noise_3_2_40k_10k / total)
accuracy_noise_3_2_30k_20k = (running_accuracy_noise_3_2_30k_20k / total)
accuracy_noise_3_2_20k_30k = (running_accuracy_noise_3_2_20k_30k / total)
accuracy_noise_3_2_10k_40k = (running_accuracy_noise_3_2_10k_40k / total)
accuracy_noise_3_2_0k_50k = (running_accuracy_noise_3_2_0k_50k / total)

accuracy_noise_3_3_50k_0k = (running_accuracy_noise_3_3_50k_0k / total)
accuracy_noise_3_3_40k_10k = (running_accuracy_noise_3_3_40k_10k / total)
accuracy_noise_3_3_30k_20k = (running_accuracy_noise_3_3_30k_20k / total)
accuracy_noise_3_3_20k_30k = (running_accuracy_noise_3_3_20k_30k / total)
accuracy_noise_3_3_10k_40k = (running_accuracy_noise_3_3_10k_40k / total)
accuracy_noise_3_3_0k_50k = (running_accuracy_noise_3_3_0k_50k / total)

accuracy_noise_3_4_50k_0k = (running_accuracy_noise_3_4_50k_0k / total)
accuracy_noise_3_4_40k_10k = (running_accuracy_noise_3_4_40k_10k / total)
accuracy_noise_3_4_30k_20k = (running_accuracy_noise_3_4_30k_20k / total)
accuracy_noise_3_4_20k_30k = (running_accuracy_noise_3_4_20k_30k / total)
accuracy_noise_3_4_10k_40k = (running_accuracy_noise_3_4_10k_40k / total)
accuracy_noise_3_4_0k_50k = (running_accuracy_noise_3_4_0k_50k / total)

accuracy_noise_3_5_50k_0k = (running_accuracy_noise_3_5_50k_0k / total)
accuracy_noise_3_5_40k_10k = (running_accuracy_noise_3_5_40k_10k / total)
accuracy_noise_3_5_30k_20k = (running_accuracy_noise_3_5_30k_20k / total)
accuracy_noise_3_5_20k_30k = (running_accuracy_noise_3_5_20k_30k / total)
accuracy_noise_3_5_10k_40k = (running_accuracy_noise_3_5_10k_40k / total)
accuracy_noise_3_5_0k_50k = (running_accuracy_noise_3_5_0k_50k / total)

accuracy_noise_3_6_50k_0k = (running_accuracy_noise_3_6_50k_0k / total)
accuracy_noise_3_6_40k_10k = (running_accuracy_noise_3_6_40k_10k / total)
accuracy_noise_3_6_30k_20k = (running_accuracy_noise_3_6_30k_20k / total)
accuracy_noise_3_6_20k_30k = (running_accuracy_noise_3_6_20k_30k / total)
accuracy_noise_3_6_10k_40k = (running_accuracy_noise_3_6_10k_40k / total)
accuracy_noise_3_6_0k_50k = (running_accuracy_noise_3_6_0k_50k / total)


accuracy_50k_0k_noise_1 = [accuracy_original_50k_0k, accuracy_noise_1_1_50k_0k, accuracy_noise_1_2_50k_0k,
                            accuracy_noise_1_3_50k_0k, accuracy_noise_1_4_50k_0k, accuracy_noise_1_5_50k_0k,
                            accuracy_noise_1_6_50k_0k]

accuracy_40k_10k_noise_1 = [accuracy_original_40k_10k, accuracy_noise_1_1_40k_10k, accuracy_noise_1_2_40k_10k,
                             accuracy_noise_1_3_40k_10k, accuracy_noise_1_4_40k_10k, accuracy_noise_1_5_40k_10k,
                             accuracy_noise_1_6_40k_10k]

accuracy_30k_20k_noise_1 = [accuracy_original_30k_20k, accuracy_noise_1_1_30k_20k, accuracy_noise_1_2_30k_20k,
                             accuracy_noise_1_3_30k_20k, accuracy_noise_1_4_30k_20k, accuracy_noise_1_5_30k_20k,
                             accuracy_noise_1_6_30k_20k]

accuracy_20k_30k_noise_1 = [accuracy_original_20k_30k, accuracy_noise_1_1_20k_30k, accuracy_noise_1_2_20k_30k,
                             accuracy_noise_1_3_20k_30k, accuracy_noise_1_4_20k_30k, accuracy_noise_1_5_20k_30k,
                             accuracy_noise_1_6_20k_30k]

accuracy_10k_40k_noise_1 = [accuracy_original_10k_40k,  accuracy_noise_1_1_10k_40k, accuracy_noise_1_2_10k_40k,
                             accuracy_noise_1_3_10k_40k, accuracy_noise_1_4_10k_40k, accuracy_noise_1_5_10k_40k,
                             accuracy_noise_1_6_10k_40k]

accuracy_0k_50k_noise_1 = [accuracy_original_0k_50k, accuracy_noise_1_1_0k_50k, accuracy_noise_1_2_0k_50k,
                            accuracy_noise_1_3_0k_50k, accuracy_noise_1_4_0k_50k, accuracy_noise_1_5_0k_50k,
                            accuracy_noise_1_6_0k_50k]


accuracy_50k_0k_noise_2 = [accuracy_original_50k_0k, accuracy_noise_2_1_50k_0k, accuracy_noise_2_2_50k_0k,
                           accuracy_noise_2_3_50k_0k, accuracy_noise_2_4_50k_0k, accuracy_noise_2_5_50k_0k,
                           accuracy_noise_2_6_50k_0k]

accuracy_40k_10k_noise_2 = [accuracy_original_40k_10k, accuracy_noise_2_1_40k_10k, accuracy_noise_2_2_40k_10k,
                            accuracy_noise_2_3_40k_10k, accuracy_noise_2_4_40k_10k, accuracy_noise_2_5_40k_10k,
                            accuracy_noise_2_6_40k_10k]

accuracy_30k_20k_noise_2 = [accuracy_original_30k_20k, accuracy_noise_2_1_30k_20k, accuracy_noise_2_2_30k_20k,
                            accuracy_noise_2_3_30k_20k, accuracy_noise_2_4_30k_20k, accuracy_noise_2_5_30k_20k,
                            accuracy_noise_2_6_30k_20k]

accuracy_20k_30k_noise_2 = [accuracy_original_20k_30k, accuracy_noise_2_1_20k_30k, accuracy_noise_2_2_20k_30k,
                            accuracy_noise_2_3_20k_30k, accuracy_noise_2_4_20k_30k, accuracy_noise_2_5_20k_30k,
                            accuracy_noise_2_6_20k_30k]

accuracy_10k_40k_noise_2 = [accuracy_original_10k_40k, accuracy_noise_2_1_10k_40k, accuracy_noise_2_2_10k_40k,
                            accuracy_noise_2_3_10k_40k, accuracy_noise_2_4_10k_40k, accuracy_noise_2_5_10k_40k,
                            accuracy_noise_2_6_10k_40k]

accuracy_0k_50k_noise_2 = [accuracy_original_0k_50k, accuracy_noise_2_1_0k_50k, accuracy_noise_2_2_0k_50k,
                           accuracy_noise_2_3_0k_50k, accuracy_noise_2_4_0k_50k, accuracy_noise_2_5_0k_50k,
                           accuracy_noise_2_6_0k_50k]


accuracy_50k_0k_noise_3 = [accuracy_original_50k_0k, accuracy_noise_3_1_50k_0k,  accuracy_noise_3_2_50k_0k,
                       accuracy_noise_3_3_50k_0k, accuracy_noise_3_4_50k_0k, accuracy_noise_3_5_50k_0k,
                       accuracy_noise_3_6_50k_0k]

accuracy_40k_10k_noise_3 = [accuracy_original_40k_10k, accuracy_noise_3_1_40k_10k,  accuracy_noise_3_2_40k_10k,
                        accuracy_noise_3_3_40k_10k, accuracy_noise_3_4_40k_10k, accuracy_noise_3_5_40k_10k,
                        accuracy_noise_3_6_40k_10k]

accuracy_30k_20k_noise_3 = [accuracy_original_30k_20k, accuracy_noise_3_1_30k_20k,  accuracy_noise_3_2_30k_20k,
                        accuracy_noise_3_3_30k_20k, accuracy_noise_3_4_30k_20k, accuracy_noise_3_5_30k_20k,
                        accuracy_noise_3_6_30k_20k]

accuracy_20k_30k_noise_3 = [accuracy_original_20k_30k, accuracy_noise_3_1_20k_30k,  accuracy_noise_3_2_20k_30k,
                        accuracy_noise_3_3_20k_30k, accuracy_noise_3_4_20k_30k, accuracy_noise_3_5_20k_30k,
                        accuracy_noise_3_6_20k_30k]

accuracy_10k_40k_noise_3 = [accuracy_original_10k_40k, accuracy_noise_3_1_10k_40k,  accuracy_noise_3_2_10k_40k,
                        accuracy_noise_3_3_10k_40k, accuracy_noise_3_4_10k_40k, accuracy_noise_3_5_10k_40k,
                        accuracy_noise_3_6_10k_40k]

accuracy_0k_50k_noise_3 = [accuracy_original_0k_50k, accuracy_noise_3_1_0k_50k,  accuracy_noise_3_2_0k_50k,
                       accuracy_noise_3_3_0k_50k, accuracy_noise_3_4_0k_50k, accuracy_noise_3_5_0k_50k,
                       accuracy_noise_3_6_0k_50k]

stats = []

stats.append('original_50k_0k: {:.4f}'.format(accuracy_original_50k_0k))
stats.append('original_40k_10k: {:.4f}'.format(accuracy_original_40k_10k))
stats.append('original_30k_20k: {:.4f}'.format(accuracy_original_30k_20k))
stats.append('original_20k_30k: {:.4f}'.format(accuracy_original_20k_30k))
stats.append('original_10k_40k: {:.4f}'.format(accuracy_original_10k_40k))
stats.append('original_0k_50k: {:.4f}'.format(accuracy_original_0k_50k))
stats.append('')
stats.append('')
stats.append('noise_1_1_50k_0k: {:.4f}'.format(accuracy_noise_1_1_50k_0k))
stats.append('noise_1_1_40k_10k: {:.4f}'.format(accuracy_noise_1_1_40k_10k))
stats.append('noise_1_1_30k_20k: {:.4f}'.format(accuracy_noise_1_1_30k_20k))
stats.append('noise_1_1_20k_30k: {:.4f}'.format(accuracy_noise_1_1_20k_30k))
stats.append('noise_1_1_10k_40k: {:.4f}'.format(accuracy_noise_1_1_10k_40k))
stats.append('noise_1_1_0k_50k: {:.4f}'.format(accuracy_noise_1_1_0k_50k))
stats.append('')
stats.append('noise_1_2_50k_0k: {:.4f}'.format(accuracy_noise_1_2_50k_0k))
stats.append('noise_1_2_40k_10k: {:.4f}'.format(accuracy_noise_1_2_40k_10k))
stats.append('noise_1_2_30k_20k: {:.4f}'.format(accuracy_noise_1_2_30k_20k))
stats.append('noise_1_2_20k_30k: {:.4f}'.format(accuracy_noise_1_2_20k_30k))
stats.append('noise_1_2_10k_40k: {:.4f}'.format(accuracy_noise_1_2_10k_40k))
stats.append('noise_1_2_0k_50k: {:.4f}'.format(accuracy_noise_1_2_0k_50k))
stats.append('')
stats.append('noise_1_3_50k_0k: {:.4f}'.format(accuracy_noise_1_3_50k_0k))
stats.append('noise_1_3_40k_10k: {:.4f}'.format(accuracy_noise_1_3_40k_10k))
stats.append('noise_1_3_30k_20k: {:.4f}'.format(accuracy_noise_1_3_30k_20k))
stats.append('noise_1_3_20k_30k: {:.4f}'.format(accuracy_noise_1_3_20k_30k))
stats.append('noise_1_3_10k_40k: {:.4f}'.format(accuracy_noise_1_3_10k_40k))
stats.append('noise_1_3_0k_50k: {:.4f}'.format(accuracy_noise_1_3_0k_50k))
stats.append('')
stats.append('noise_1_4_50k_0k: {:.4f}'.format(accuracy_noise_1_4_50k_0k))
stats.append('noise_1_4_40k_10k: {:.4f}'.format(accuracy_noise_1_4_40k_10k))
stats.append('noise_1_4_30k_20k: {:.4f}'.format(accuracy_noise_1_4_30k_20k))
stats.append('noise_1_4_20k_30k: {:.4f}'.format(accuracy_noise_1_4_20k_30k))
stats.append('noise_1_4_10k_40k: {:.4f}'.format(accuracy_noise_1_4_10k_40k))
stats.append('noise_1_4_0k_50k: {:.4f}'.format(accuracy_noise_1_4_0k_50k))
stats.append('')
stats.append('noise_1_5_50k_0k: {:.4f}'.format(accuracy_noise_1_5_50k_0k))
stats.append('noise_1_5_40k_10k: {:.4f}'.format(accuracy_noise_1_5_40k_10k))
stats.append('noise_1_5_30k_20k: {:.4f}'.format(accuracy_noise_1_5_30k_20k))
stats.append('noise_1_5_20k_30k: {:.4f}'.format(accuracy_noise_1_5_20k_30k))
stats.append('noise_1_5_10k_40k: {:.4f}'.format(accuracy_noise_1_5_10k_40k))
stats.append('noise_1_5_0k_50k: {:.4f}'.format(accuracy_noise_1_5_0k_50k))
stats.append('')
stats.append('noise_1_6_50k_0k: {:.4f}'.format(accuracy_noise_1_6_50k_0k))
stats.append('noise_1_6_40k_10k: {:.4f}'.format(accuracy_noise_1_6_40k_10k))
stats.append('noise_1_6_30k_20k: {:.4f}'.format(accuracy_noise_1_6_30k_20k))
stats.append('noise_1_6_20k_30k: {:.4f}'.format(accuracy_noise_1_6_20k_30k))
stats.append('noise_1_6_10k_40k: {:.4f}'.format(accuracy_noise_1_6_10k_40k))
stats.append('noise_1_6_0k_50k: {:.4f}'.format(accuracy_noise_1_6_0k_50k))
stats.append('')
stats.append('')
stats.append('noise_2_1_50k_0k: {:.4f}'.format(accuracy_noise_2_1_50k_0k))
stats.append('noise_2_1_40k_10k: {:.4f}'.format(accuracy_noise_2_1_40k_10k))
stats.append('noise_2_1_30k_20k: {:.4f}'.format(accuracy_noise_2_1_30k_20k))
stats.append('noise_2_1_20k_30k: {:.4f}'.format(accuracy_noise_2_1_20k_30k))
stats.append('noise_2_1_10k_40k: {:.4f}'.format(accuracy_noise_2_1_10k_40k))
stats.append('noise_2_1_0k_50k: {:.4f}'.format(accuracy_noise_2_1_0k_50k))
stats.append('')
stats.append('noise_2_2_50k_0k: {:.4f}'.format(accuracy_noise_2_2_50k_0k))
stats.append('noise_2_2_40k_10k: {:.4f}'.format(accuracy_noise_2_2_40k_10k))
stats.append('noise_2_2_30k_20k: {:.4f}'.format(accuracy_noise_2_2_30k_20k))
stats.append('noise_2_2_20k_30k: {:.4f}'.format(accuracy_noise_2_2_20k_30k))
stats.append('noise_2_2_10k_40k: {:.4f}'.format(accuracy_noise_2_2_10k_40k))
stats.append('noise_2_2_0k_50k: {:.4f}'.format(accuracy_noise_2_2_0k_50k))
stats.append('')
stats.append('noise_2_3_50k_0k: {:.4f}'.format(accuracy_noise_2_3_50k_0k))
stats.append('noise_2_3_40k_10k: {:.4f}'.format(accuracy_noise_2_3_40k_10k))
stats.append('noise_2_3_30k_20k: {:.4f}'.format(accuracy_noise_2_3_30k_20k))
stats.append('noise_2_3_20k_30k: {:.4f}'.format(accuracy_noise_2_3_20k_30k))
stats.append('noise_2_3_10k_40k: {:.4f}'.format(accuracy_noise_2_3_10k_40k))
stats.append('noise_2_3_0k_50k: {:.4f}'.format(accuracy_noise_2_3_0k_50k))
stats.append('')
stats.append('noise_2_4_50k_0k: {:.4f}'.format(accuracy_noise_2_4_50k_0k))
stats.append('noise_2_4_40k_10k: {:.4f}'.format(accuracy_noise_2_4_40k_10k))
stats.append('noise_2_4_30k_20k: {:.4f}'.format(accuracy_noise_2_4_30k_20k))
stats.append('noise_2_4_20k_30k: {:.4f}'.format(accuracy_noise_2_4_20k_30k))
stats.append('noise_2_4_10k_40k: {:.4f}'.format(accuracy_noise_2_4_10k_40k))
stats.append('noise_2_4_0k_50k: {:.4f}'.format(accuracy_noise_2_4_0k_50k))
stats.append('')
stats.append('noise_2_5_50k_0k: {:.4f}'.format(accuracy_noise_2_5_50k_0k))
stats.append('noise_2_5_40k_10k: {:.4f}'.format(accuracy_noise_2_5_40k_10k))
stats.append('noise_2_5_30k_20k: {:.4f}'.format(accuracy_noise_2_5_30k_20k))
stats.append('noise_2_5_20k_30k: {:.4f}'.format(accuracy_noise_2_5_20k_30k))
stats.append('noise_2_5_10k_40k: {:.4f}'.format(accuracy_noise_2_5_10k_40k))
stats.append('noise_2_5_0k_50k: {:.4f}'.format(accuracy_noise_2_5_0k_50k))
stats.append('')
stats.append('noise_2_6_50k_0k: {:.4f}'.format(accuracy_noise_2_6_50k_0k))
stats.append('noise_2_6_40k_10k: {:.4f}'.format(accuracy_noise_2_6_40k_10k))
stats.append('noise_2_6_30k_20k: {:.4f}'.format(accuracy_noise_2_6_30k_20k))
stats.append('noise_2_6_20k_30k: {:.4f}'.format(accuracy_noise_2_6_20k_30k))
stats.append('noise_2_6_10k_40k: {:.4f}'.format(accuracy_noise_2_6_10k_40k))
stats.append('noise_2_6_0k_50k: {:.4f}'.format(accuracy_noise_2_6_0k_50k))
stats.append('')
stats.append('')
stats.append('noise_3_1_50k_0k: {:.4f}'.format(accuracy_noise_3_1_50k_0k))
stats.append('noise_3_1_40k_10k: {:.4f}'.format(accuracy_noise_3_1_40k_10k))
stats.append('noise_3_1_30k_20k: {:.4f}'.format(accuracy_noise_3_1_30k_20k))
stats.append('noise_3_1_20k_30k: {:.4f}'.format(accuracy_noise_3_1_20k_30k))
stats.append('noise_3_1_10k_40k: {:.4f}'.format(accuracy_noise_3_1_10k_40k))
stats.append('noise_3_1_0k_50k: {:.4f}'.format(accuracy_noise_3_1_0k_50k))
stats.append('')
stats.append('noise_3_2_50k_0k: {:.4f}'.format(accuracy_noise_3_2_50k_0k))
stats.append('noise_3_2_40k_10k: {:.4f}'.format(accuracy_noise_3_2_40k_10k))
stats.append('noise_3_2_30k_20k: {:.4f}'.format(accuracy_noise_3_2_30k_20k))
stats.append('noise_3_2_20k_30k: {:.4f}'.format(accuracy_noise_3_2_20k_30k))
stats.append('noise_3_2_10k_40k: {:.4f}'.format(accuracy_noise_3_2_10k_40k))
stats.append('noise_3_2_0k_50k: {:.4f}'.format(accuracy_noise_3_2_0k_50k))
stats.append('')
stats.append('noise_3_3_50k_0k: {:.4f}'.format(accuracy_noise_3_3_50k_0k))
stats.append('noise_3_3_40k_10k: {:.4f}'.format(accuracy_noise_3_3_40k_10k))
stats.append('noise_3_3_30k_20k: {:.4f}'.format(accuracy_noise_3_3_30k_20k))
stats.append('noise_3_3_20k_30k: {:.4f}'.format(accuracy_noise_3_3_20k_30k))
stats.append('noise_3_3_10k_40k: {:.4f}'.format(accuracy_noise_3_3_10k_40k))
stats.append('noise_3_3_0k_50k: {:.4f}'.format(accuracy_noise_3_3_0k_50k))
stats.append('')
stats.append('noise_3_4_50k_0k: {:.4f}'.format(accuracy_noise_3_4_50k_0k))
stats.append('noise_3_4_40k_10k: {:.4f}'.format(accuracy_noise_3_4_40k_10k))
stats.append('noise_3_4_30k_20k: {:.4f}'.format(accuracy_noise_3_4_30k_20k))
stats.append('noise_3_4_20k_30k: {:.4f}'.format(accuracy_noise_3_4_20k_30k))
stats.append('noise_3_4_10k_40k: {:.4f}'.format(accuracy_noise_3_4_10k_40k))
stats.append('noise_3_4_0k_50k: {:.4f}'.format(accuracy_noise_3_4_0k_50k))
stats.append('')
stats.append('noise_3_5_50k_0k: {:.4f}'.format(accuracy_noise_3_5_50k_0k))
stats.append('noise_3_5_40k_10k: {:.4f}'.format(accuracy_noise_3_5_40k_10k))
stats.append('noise_3_5_30k_20k: {:.4f}'.format(accuracy_noise_3_5_30k_20k))
stats.append('noise_3_5_20k_30k: {:.4f}'.format(accuracy_noise_3_5_20k_30k))
stats.append('noise_3_5_10k_40k: {:.4f}'.format(accuracy_noise_3_5_10k_40k))
stats.append('noise_3_5_0k_50k: {:.4f}'.format(accuracy_noise_3_5_0k_50k))
stats.append('')
stats.append('noise_3_6_50k_0k: {:.4f}'.format(accuracy_noise_3_6_50k_0k))
stats.append('noise_3_6_40k_10k: {:.4f}'.format(accuracy_noise_3_6_40k_10k))
stats.append('noise_3_6_30k_20k: {:.4f}'.format(accuracy_noise_3_6_30k_20k))
stats.append('noise_3_6_20k_30k: {:.4f}'.format(accuracy_noise_3_6_20k_30k))
stats.append('noise_3_6_10k_40k: {:.4f}'.format(accuracy_noise_3_6_10k_40k))
stats.append('noise_3_6_0k_50k: {:.4f}'.format(accuracy_noise_3_6_0k_50k))
stats.append('')
stats.append('')
stats.append('accuracy_50k_0k_noise_1: {}'.format(accuracy_50k_0k_noise_1))
stats.append('accuracy_40k_10k_noise_1: {}'.format(accuracy_40k_10k_noise_1))
stats.append('accuracy_30k_20k_noise_1: {}'.format(accuracy_30k_20k_noise_1))
stats.append('accuracy_20k_30k_noise_1: {}'.format(accuracy_20k_30k_noise_1))
stats.append('accuracy_10k_40k_noise_1: {}'.format(accuracy_10k_40k_noise_1))
stats.append('accuracy_0k_50k_noise_1: {}'.format(accuracy_0k_50k_noise_1))
stats.append('')
stats.append('accuracy_50k_0k_noise_2: {}'.format(accuracy_50k_0k_noise_2))
stats.append('accuracy_40k_10k_noise_2: {}'.format(accuracy_40k_10k_noise_2))
stats.append('accuracy_30k_20k_noise_2: {}'.format(accuracy_30k_20k_noise_2))
stats.append('accuracy_20k_30k_noise_2: {}'.format(accuracy_20k_30k_noise_2))
stats.append('accuracy_10k_40k_noise_2: {}'.format(accuracy_10k_40k_noise_2))
stats.append('accuracy_0k_50k_noise_2: {}'.format(accuracy_0k_50k_noise_2))
stats.append('')
stats.append('accuracy_50k_0k_noise_3: {}'.format(accuracy_50k_0k_noise_3))
stats.append('accuracy_40k_10k_noise_3: {}'.format(accuracy_40k_10k_noise_3))
stats.append('accuracy_30k_20k_noise_3: {}'.format(accuracy_30k_20k_noise_3))
stats.append('accuracy_20k_30k_noise_3: {}'.format(accuracy_20k_30k_noise_3))
stats.append('accuracy_10k_40k_noise_3: {}'.format(accuracy_10k_40k_noise_3))
stats.append('accuracy_0k_50k_noise_3: {}'.format(accuracy_0k_50k_noise_3))

with open(r'results.txt', 'w') as fp:
    for parameter in stats:
        fp.write('{}\n'.format(parameter))


x_noise_1 = x_noise_2 = x_noise_3 = ['0', '1', '2', '3', '4', '5', '6']


fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.subplots_adjust(hspace=0.3)
fig.set_size_inches(8, 15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax1.plot(x_noise_1, accuracy_50k_0k_noise_1, label='50k original / 0 dream', color='#EC6061')
ax1.plot(x_noise_1, accuracy_40k_10k_noise_1, label='40k original / 10k dream', color='#DF7C72')
ax1.plot(x_noise_1, accuracy_30k_20k_noise_1, label='30k original / 20k dream', color='#D09383')
ax1.plot(x_noise_1, accuracy_20k_30k_noise_1, label='20k original / 30k dream', color='#BFA694')
ax1.plot(x_noise_1, accuracy_10k_40k_noise_1, label='10k original / 40k dream', color='#A9B4A3')
ax1.plot(x_noise_1, accuracy_0k_50k_noise_1, label='0 original / 50k dream', color='#8CBEB2')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('gaussian blur intensity')
ax1.legend(frameon = False)

ax2.plot(x_noise_2, accuracy_50k_0k_noise_2, label='50k original / 0 dream', color='#EC6061')
ax2.plot(x_noise_2, accuracy_40k_10k_noise_2, label='40k original / 10k dream', color='#DF7C72')
ax2.plot(x_noise_2, accuracy_30k_20k_noise_2, label='30k original / 20k dream', color='#D09383')
ax2.plot(x_noise_2, accuracy_20k_30k_noise_2, label='20k original / 30k dream', color='#BFA694')
ax2.plot(x_noise_2, accuracy_10k_40k_noise_2, label='10k original / 40k dream', color='#A9B4A3')
ax2.plot(x_noise_2, accuracy_0k_50k_noise_2, label='0 original / 50k dream', color='#8CBEB2')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('contrast decrease intensity')
ax2.legend(frameon = False)

ax3.plot(x_noise_3, accuracy_50k_0k_noise_3, label='50k original / 0 dream', color='#EC6061')
ax3.plot(x_noise_3, accuracy_40k_10k_noise_3, label='40k original / 10k dream', color='#DF7C72')
ax3.plot(x_noise_3, accuracy_30k_20k_noise_3, label='30k original / 20k dream', color='#D09383')
ax3.plot(x_noise_3, accuracy_20k_30k_noise_3, label='20k original / 30k dream', color='#BFA694')
ax3.plot(x_noise_3, accuracy_10k_40k_noise_3, label='10k original / 40k dream', color='#A9B4A3')
ax3.plot(x_noise_3, accuracy_0k_50k_noise_3, label='0 original / 50k dream', color='#8CBEB2')
ax3.set_ylabel('accuracy')
ax3.set_xlabel('high-pass filter intensity')
ax3.legend(frameon = False)

plt.savefig('plot results.pdf', facecolor='w', edgecolor='w',
        orientation='portrait', bbox_inches=None, pad_inches=0.1)
