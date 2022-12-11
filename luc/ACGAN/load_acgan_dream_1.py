import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

device = 'cpu'

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_one_hot = F.one_hot(torch.arange(0, 10), num_classes=len(classes))

loaded_epoch = 500
n_images = 30000
latent_size = 100
n_classes = len(classes)
filter_size_g = 96


class Generator(nn.Module):

    def __init__(self, latent_size, nb_filter, n_classes):
        super(Generator, self).__init__()

        self.embedding = nn.Linear(n_classes, latent_size)

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(latent_size, nb_filter * 8, 4, 1, 0, bias=False),
                                    nn.ReLU(True))

        self.layer2 = nn.Sequential(nn.ConvTranspose2d(nb_filter * 8, nb_filter * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 4),
                                    nn.ReLU(True))

        self.layer3 = nn.Sequential(nn.ConvTranspose2d(nb_filter * 4, nb_filter * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 2),
                                    nn.ReLU(True))

        self.layer4 = nn.Sequential(nn.ConvTranspose2d(nb_filter * 2, nb_filter, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter),
                                    nn.ReLU(True))

        self.layer5 = nn.Sequential(nn.ConvTranspose2d(nb_filter, 3, 4, 2, 1, bias=False),
                                    nn.Tanh())

    def forward(self, latent, label):
        label_embedding = self.embedding(label)
        x = torch.mul(label_embedding, latent)
        x = x.view(x.size(0), -1, 1, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


def weighing(label_one_hot, weight):
    label_one_hot = classes_one_hot[label_one_hot].to(device)
    label_one_hot = label_one_hot.tolist()
    label_one_hot = [item for elem in label_one_hot for item in elem]

    for i, value in enumerate(label_one_hot):
        if label_one_hot[i] == 1:
            label_one_hot[i] = weight

    label_one_hot = torch.FloatTensor(label_one_hot).to(device)

    return label_one_hot


G = Generator(latent_size, filter_size_g, n_classes).to(device)
G.load_state_dict(torch.load('checkpoint epoch {}.pt'.format(loaded_epoch), map_location=device))
G.eval()

transform_PIL = transforms.ToPILImage()

# generate the images
for i in range(n_images):
    label = torch.LongTensor(np.random.randint(0, 10, 1))

    label_one_hot = weighing(label, 1).to(device)
    label_one_hot = label_one_hot.view(1, 10)

    latent_value = torch.randn(1, 100).to(device)

    img = G(latent_value, label_one_hot)

    save_image(img, './samples_dreamed_30k_0k_no_interpolation/{} {}.png'.format(classes[label], i), padding=2,
               normalize=True)
