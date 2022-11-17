import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.utils as vutils

device = 'cpu'


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_one_hot = F.one_hot(torch.arange(0, 10), num_classes=len(classes))

latent_size = 100
n_classes = len(classes)
loaded_epoch = 500
filter_size_g = 96
interpolation_steps = 100


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


G = Generator(latent_size, filter_size_g, n_classes).to(device)
G.load_state_dict(torch.load('checkpoints/checkpoint epoch {}.pt'.format(loaded_epoch), map_location=device))
G.eval()


fixed_latent = torch.randn(100, 100, device=device)

for interpolation in range(interpolation_steps+1):

    fixed_labels = torch.zeros(100, 10, device=device)

    if interpolation == 0:
        w_1 = 0
        w_2 = 1

    else:
        w_1 += 1 / interpolation_steps
        w_2 = 1 - w_1


    for i in range(10):
        for j in range(10):
            fixed_labels[j*10+i][j] = w_2

    for i in range(10):
        for j in range(10):
            fixed_labels[i*10+j][j] += w_1


    fixed_labels = fixed_labels.view(100, 10)

    img_list = []
    transform_PIL = transforms.ToPILImage()


    with torch.no_grad():
        img = G(fixed_latent, fixed_labels).detach().cpu()
        img_list.append(vutils.make_grid(torch.reshape(img,(100, 3, 64, 64)), nrow=10, padding=2, normalize=True))
        transform_PIL(img_list[-1]).save('grid interpolation/{}_{:.3f}_{:.3f}.png'.format(interpolation, w_1, w_2))
