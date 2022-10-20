import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
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

    def __init__(self , nb_filter, n_classes):
        super(Generator, self).__init__()
        self.fc_layer = nn.Linear(latent_size+n_classes, 110)
        # self.label_embedding = nn.Embedding(n_classes, latent_size)
        # self.conv1 = nn.ConvTranspose2d(nb_filter * 8, nb_filter * 4, 1, 0)
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
        # x = torch.mul(self.label_embedding(cl), input)
        # x = x.view(x.size(0), -1, 1, 1)

        ## print(input.size()) returns torch.Size([100, 110]), it was torch.Size([100, 100])

        x = self.fc_layer(input)
        x = x.view(x.size(0), -1, 1, 1)
        ## print(x.size() returns torch.Size([100, 512, 1, 1]), it should be torch.Size([100, 100, 1, 1])

        x = self.conv1(x)
        ## print(x.size()) returns torch.Size([100, 512, 2, 2]), it should be torch.Size([100, 512, 4, 4])

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
        ## print(x.size()) returns torch.Size([100, 3, 32, 32]), it should be torch.Size([100, 3, 64, 64])
        return torch.tanh(x)

G = Generator(64, n_classes).to(device)
G.load_state_dict(torch.load('/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/training 6/checkpoints/checkpoint epoch 1.pt'))
G.eval()

n_images = 10

fixed_latent = torch.randn(100, 100, device=device)
fixed_labels = torch.zeros(100, 10, device=device)

for i in range(10):
     for j in range(10):
         fixed_labels[j*10+i][j] = 0.7

for i in range(10):
    for j in range(10):
        fixed_labels[i*10+j][j] += 0.3

fixed_labels = fixed_labels.view(100, 10)

fixed_noise = torch.cat((fixed_latent,fixed_labels), 1)


img_list = []
transform_PIL = transforms.ToPILImage()

with torch.no_grad():
    img = G(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(torch.reshape(img,(100,c_dim,in_h,in_w)),nrow=10, padding=2, normalize=True))
    transform_PIL(img_list[-1]).save('/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/training 6/samples dreamed/grid.png')
