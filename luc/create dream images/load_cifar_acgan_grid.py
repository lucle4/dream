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
loaded_epoch = 20

w_1 = 0.3
w_2 = 1 - w_1


class Generator(nn.Module):

    def __init__(self, latent_size, nb_filter, n_classes):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_size+n_classes, nb_filter * 16)
        self.conv1 = nn.ConvTranspose2d(nb_filter * 16, nb_filter * 8, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(nb_filter * 8)
        self.conv2 = nn.ConvTranspose2d(nb_filter * 8, nb_filter * 4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(nb_filter * 4)
        self.conv3 = nn.ConvTranspose2d(nb_filter * 4, nb_filter * 2, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(nb_filter * 2)
        self.conv4 = nn.ConvTranspose2d(nb_filter * 2, nb_filter * 1, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(nb_filter * 1)
        self.conv5 = nn.ConvTranspose2d(nb_filter * 1, 3, 4, 2, 1)

    def forward(self, latent):
        x = self.fc1(latent)
        x = x.view(x.size(0), -1, 1, 1)
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


G = Generator(latent_size, 64, n_classes).to(device)
G.load_state_dict(torch.load('/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/trainings/training 9/checkpoints/checkpoint epoch {}.pt'.format(loaded_epoch)))
G.eval()


fixed_latent = torch.randn(100, 100, device=device)
fixed_labels = torch.zeros(100, 10, device=device)


for i in range(10):
     for j in range(10):
         fixed_labels[j*10+i][j] = w_2

for i in range(10):
    for j in range(10):
        fixed_labels[i*10+j][j] += w_1


fixed_labels = fixed_labels.view(100, 10)

fixed_noise = torch.cat((fixed_latent,fixed_labels), 1)

img_list = []
transform_PIL = transforms.ToPILImage()


with torch.no_grad():
    img = G(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(torch.reshape(img,(100,3,64,64)),nrow=10, padding=2, normalize=True))
    transform_PIL(img_list[-1]).save('/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/trainings/training 9/dream grid epoch {}.png'.format(loaded_epoch))
