import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import csv
import os

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_classes = len(classes)
size_dataset = 10000  # max 50000

classes_one_hot = F.one_hot(torch.arange(0, 10), n_classes)

directory = os.getcwd()

img_dir = os.path.join(directory, 'samples_dreamed_50k_no_interpolation')
csv_dir = os.path.join(directory, 'samples_dreamed_50k_no_interpolation.csv')

label_list = []
img_list = []

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True)

for i, (input, target) in enumerate(train_loader):
    if i == size_dataset:
        break

    target = int(target)
    label_one_hot = classes_one_hot[target]
    label_one_hot = label_one_hot.tolist()
    label_one_hot = ', '.join(map(str, label_one_hot))

    label_list.append(label_one_hot)
    img_list.append('{} {}.png'.format(classes[target], i + 1))

    save_image(input, 'samples_original_10k/{} {}.png'.format(classes[target], i + 1))

list_csv = [list(i) for i in zip(img_list, label_list)]

with open(csv_dir, 'w') as file:
    write = csv.writer(file)
    write.writerows(list_csv)
