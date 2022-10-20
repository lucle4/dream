import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import math

device = 'cpu'

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_one_hot = F.one_hot(torch.arange(0, 10), num_classes=len(classes))

latent_size = 100
n_classes = len(classes)

w_1 = 0.3
w_2 = 1 - w_1

transform_PIL=transforms.ToPILImage()

class Generator(nn.Module):

    def __init__(self , nb_filter, n_classes):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(110, nb_filter * 8, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(nb_filter * 8)
        self.conv2 = nn.ConvTranspose2d(nb_filter * 8, nb_filter * 4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(nb_filter * 4)
        self.conv3 = nn.ConvTranspose2d(nb_filter * 4, nb_filter * 2, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(nb_filter * 2)
        self.conv4 = nn.ConvTranspose2d(nb_filter * 2, nb_filter * 1, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(nb_filter * 1)
        self.conv5 = nn.ConvTranspose2d(nb_filter * 1, 3, 4, 2, 1)
        self.__initialize_weights()

    def forward(self, input):
        x = input.view(input.size(0), -1, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        return torch.tanh(x)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

G = Generator(64, n_classes).to(device)
G.load_state_dict(torch.load('/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/training 6/checkpoints/checkpoint epoch 1.pt'))
G.eval()

n_images = 10

def weighing(label_one_hot, weight):

    label_one_hot = classes_one_hot[label_one_hot]
    label_one_hot = label_one_hot.tolist()
    label_one_hot = [item for elem in label_one_hot for item in elem]

    for i, value in enumerate(label_one_hot):
        if label_one_hot[i] == 1:
            label_one_hot[i] = weight
    label_one_hot = torch.FloatTensor(label_one_hot)

    return(label_one_hot)

# generate the images
for i in range(n_images):
    label_1 = torch.LongTensor(np.random.randint(0, 10, 1)).to(device)
    label_2 = torch.LongTensor(np.random.randint(0, 10, 1)).to(device)

    # different weight for label 1
    while label_1 == label_2:
        label_2 = torch.LongTensor(np.random.randint(0, 10, 1)).to(device)

    label_1_one_hot = weighing(label_1, w_1)
    label_2_one_hot = weighing(label_2, w_2)

    label_combined = label_1_one_hot + label_2_one_hot
    label_combined = label_combined.view(1, 10)

    latent_value = torch.randn(1, 100).to(device)
    latent = torch.cat((latent_value,label_combined),dim=1)

    img = G(latent)

    save_image(img, '/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/training 6/samples dreamed/{}_{}_{}_{} {}.png'.format(classes[label_1], w_1, classes[label_2], w_2, i), padding=2, normalize = True)
