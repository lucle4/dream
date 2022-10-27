import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

device = 'cpu'


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

n_epochs = 50
batch_size = 100
latent_size = 100
n_classes = len(classes)
filter_size = 64
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def compute_cls_acc(predictLabel, target):
    return ((predictLabel.argmax(dim=1) == target)*1.0).sum()


img_list = []

fixed_latent = torch.randn(100, latent_size, device=device)
fixed_labels = torch.zeros(100, n_classes, device=device)

fixed_input = torch.cat((fixed_latent, fixed_labels), dim=1)

for j in range(10):
	for i in range(n_classes):
		fixed_labels[i*10+j][i] = 1


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
        self.__initialize_weights()

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

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class Discriminator(nn.Module):

    def __init__(self, nb_filter, n_classes):
        super(Discriminator, self).__init__()
        self.nb_filter = nb_filter
        self.conv1 = nn.Conv2d(3, nb_filter, 4, 2, 1)
        self.conv2 = nn.Conv2d(nb_filter, nb_filter * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(nb_filter * 2)
        self.conv3 = nn.Conv2d(nb_filter * 2, nb_filter * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(nb_filter * 4)
        self.conv4 = nn.Conv2d(nb_filter * 4, nb_filter * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(nb_filter * 8)
        self.conv5 = nn.Conv2d(nb_filter * 8, nb_filter * 1, 4, 1, 0)
        self.gan_linear = nn.Linear(nb_filter * 1, 1)
        self.aux_linear = nn.Linear(nb_filter * 1, n_classes)
        self.__initialize_weights()

    def forward(self, input):
        x = self.conv1(input)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv5(x)
        x = x.view(-1, self.nb_filter * 1)
        c = self.aux_linear(x)
        s = self.gan_linear(x)
        s = torch.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

G = Generator(latent_size, filter_size, n_classes).to(device)
D = Discriminator(filter_size, n_classes).to(device)

optimizerG = torch.optim.Adam(G.parameters(), lr, betas = (beta_1, beta_2))
optimizerD = torch.optim.Adam(D.parameters(), lr, betas = (beta_1, beta_2))

criterion_adv = nn.BCELoss()
criterion_aux = nn.CrossEntropyLoss()

total_step = len(train_loader)


for epoch in range(n_epochs):
    for i, (input, target) in enumerate(train_loader):
        images = input.to(device)

        current_batch_size = images.size()[0]

        realLabel = torch.ones(current_batch_size).to(device)
        fakeLabel = torch.zeros(current_batch_size).to(device)

        target = torch.LongTensor(target).to(device)

        ###########
        # TRAIN D #
        ###########

        # on real data
        predictR, predictRLabel = D(images)
        loss_real_adv = criterion_adv(predictR, realLabel)
        loss_real_aux = criterion_aux(predictRLabel, target)

        real_cls_acc = compute_cls_acc(predictRLabel, target)
        real_score = predictR

        # on fake data
        latent_value = torch.randn(current_batch_size, latent_size).to(device)

        gen_labels = torch.LongTensor(np.random.randint(0, n_classes, current_batch_size)).to(device)
        cls_one_hot = torch.zeros(current_batch_size, n_classes, device=device)
        cls_one_hot[torch.arange(current_batch_size), target] = 1.0

        latent = torch.cat((latent_value, cls_one_hot),dim=1)

        fake_images= G(latent)

        predictF, predictFLabel = D(fake_images)
        loss_fake_adv = criterion_adv(predictF, fakeLabel)
        loss_fake_aux = criterion_aux(predictFLabel, gen_labels)

        fake_cls_acc = compute_cls_acc(predictFLabel, target)
        fake_score = predictF


        lossD = loss_real_adv + loss_real_aux + loss_fake_adv + loss_fake_aux

        optimizerD.zero_grad()
        optimizerG.zero_grad()
        lossD.backward()
        optimizerD.step()

        ###########
        # TRAIN G #
        ###########

        latent_value = torch.randn(current_batch_size, latent_size).to(device)

        gen_labels = torch.LongTensor(np.random.randint(0, n_classes, current_batch_size)).to(device)
        cls_one_hot = torch.zeros(current_batch_size, n_classes, device=device)
        cls_one_hot[torch.arange(current_batch_size), target] = 1.0

        latent = torch.cat((latent_value, cls_one_hot),dim=1)

        fake_images= G(latent)

        predictF, predictFLabel = D(fake_images)
        lossG_adv = criterion_adv(predictF, realLabel)
        lossG_aux = criterion_aux(predictFLabel, gen_labels)


        lossG = lossG_adv + lossG_aux

        optimizerD.zero_grad()
        optimizerG.zero_grad()
        lossG.backward()
        optimizerG.step()


        if (i+1) % 50 == 0:
            print('epoch: {}/{} batch: {}/{} G loss: {:.4f} D loss: {:.4f} Loss cls fake: {:.4f} Loss cls real: {:.4f} fake acc: {}% real acc: {}%'.format(
                epoch+1, n_epochs, i+1, total_step, lossG.item(), lossD.item(), loss_fake_aux, loss_real_aux, fake_cls_acc, real_cls_acc))

        if (i+1) % 100 == 0:
            with torch.no_grad():
                fake = G(fixed_input).detach().cpu()
                transform_PIL = transforms.ToPILImage()
                img_list.append(vutils.make_grid(torch.reshape(fake, (100, 3, 64, 64)), nrow=10, normalize=True))
                transform_PIL(img_list[-1]).save('samples/epoch {} step {}.png'.format(epoch+1, i+1))

    if (epoch+1) % 10 == 0:
        torch.save(G.state_dict(),'checkpoints/checkpoint epoch {}.pt'.format(epoch+1))


print('finished training')
