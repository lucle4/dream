import os
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, DataLoader
from torchvision.models import resnet50
from torchvision.io import read_image
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

n_epochs = 100
batch_size = 128
n_classes = len(classes)
lr = 0.1
momentum = 0.9
weight_decay = 0.0005


directory = os.getcwd()

img_dir_original = os.path.join(directory, 'original dataset/samples_original_20k')
label_dir_original = os.path.join(directory, 'original dataset/original_dataset_20k.csv')

img_dir_dream_1 = os.path.join(directory, 'dream 1 dataset/samples_dream_1_30k')
label_dir_dream_1 = os.path.join(directory, 'dream 1 dataset/dream_1_dataset_30k.csv')


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


transform_train = transforms.Compose([
    transforms.Resize(64),
    transforms.ConvertImageDtype(float),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


original_dataset = Dataset(label_dir_original, img_dir_original, transform=transform_train)

dream_1_dataset = Dataset(label_dir_dream_1, img_dir_dream_1, transform=transform_train)

combined_dataset = ConcatDataset([original_dataset, dream_1_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

if len(combined_dataset) < 50000:
    print('dataset consists of only {} images'.format(len(combined_dataset)))

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = resnet50(weights=None)
model.fc = nn.Linear(2048, 10)

if torch.cuda.is_available():
    model.cuda()


criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)


stats = []
best_accuracy = 0.0

for epoch in range (n_epochs):

    running_accuracy = 0.0
    running_test_loss = 0.0
    total = 0

    for i, (images, labels) in enumerate(combined_loader):
        model.train()

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images.float())
        train_loss = criterion(output, labels)

        train_loss.backward()
        optimizer.step()


    with torch.no_grad():
        model.eval()
        for i, (images, labels) in enumerate(test_loader):

            current_batch_size = images.size()[0]

            images = images.to(device)
            labels = labels.to(device)

            cls_one_hot = torch.zeros(current_batch_size, n_classes, device=device)
            cls_one_hot[torch.arange(current_batch_size), labels] = 1.0

            output = model(images.float())
            test_loss = criterion(output, cls_one_hot)

            _, predicted = torch.max(output, 1)

            running_test_loss += test_loss.item()
            total += labels.size(0)
            running_accuracy += (predicted == labels).sum().item()

    test_loss_epoch = running_test_loss/len(test_loader)

    accuracy = (100 * running_accuracy / total)

    if accuracy > best_accuracy and epoch > 50:
        best_accuracy = accuracy
        torch.save(model.state_dict(),'checkpoints_20k_30k/checkpoint epoch {}.pt'.format(epoch+1))

    scheduler.step()

    stats_epoch = 'epoch: {}/{} train loss: {:.4f}, test loss: {:.4f} test accuracy {}%'.format(epoch+1, n_epochs, train_loss.item(), test_loss_epoch, accuracy)
    stats.append(stats_epoch)

    with open(r'stats_20k_30k.txt', 'w') as fp:
        for parameter in stats:
            fp.write('{}\n'.format(parameter))


torch.save(model.state_dict(),'checkpoints_20k_30k/checkpoint epoch {}.pt'.format(n_epochs))
