import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

n_epochs = 500
batch_size = 100
latent_size = 100
n_classes = len(classes)
filter_size_g = 96
filter_size_d = 64
lr_g = 0.0002
lr_d = 0.0002
beta_1 = 0.5
beta_2 = 0.999


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def compute_cls_acc(predictLabel, target):
    return (((predictLabel.argmax(dim=1) == target) / batch_size) * 100).sum()


img_list = []

fixed_latent = torch.randn(100, latent_size, device=device)
fixed_labels = torch.zeros(100, n_classes, device=device)

for j in range(10):
    for i in range(n_classes):
        fixed_labels[i * 10 + j][i] = 1


class Generator(nn.Module):

    def __init__(self, latent_size, nb_filter, n_classes):
        super(Generator, self).__init__()

        self.embedding = nn.Linear(n_classes, latent_size)

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(latent_size, nb_filter * 8, 4, 1, 0, bias=False),
                                    nn.ReLU(True)
                                    )

        self.layer2 = nn.Sequential(nn.ConvTranspose2d(nb_filter * 8, nb_filter * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 4),
                                    nn.ReLU(True)
                                    )

        self.layer3 = nn.Sequential(nn.ConvTranspose2d(nb_filter * 4, nb_filter * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 2),
                                    nn.ReLU(True)
                                    )

        self.layer4 = nn.Sequential(nn.ConvTranspose2d(nb_filter * 2, nb_filter, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter),
                                    nn.ReLU(True)
                                    )

        self.layer5 = nn.Sequential(nn.ConvTranspose2d(nb_filter, 3, 4, 2, 1, bias=False),
                                    nn.Tanh()
                                    )

        self.__initialize_weights()

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

        self.layer1 = nn.Sequential(nn.Conv2d(3, nb_filter, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(nb_filter, nb_filter * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 2),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(nb_filter * 2, nb_filter * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 4),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(nb_filter * 4, nb_filter * 8, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(nb_filter * 8),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5)
                                    )

        self.adv = nn.Sequential(nn.Conv2d(nb_filter * 8, 1, 4, 1, 0, bias=False),
                                 nn.Sigmoid()
                                 )

        self.aux = nn.Sequential(nn.Conv2d(nb_filter * 8, n_classes + 1, 4, 1, 0, bias=False),
                                 nn.LogSoftmax(dim=1)
                                 )

        self.__initialize_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        adv = self.adv(x)
        aux = self.aux(x)

        adv = adv.view(-1)
        aux = aux.view(-1, 11)

        return adv, aux

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

G = Generator(latent_size, filter_size_g, n_classes).to(device)
D = Discriminator(filter_size_d, n_classes).to(device)

optimizerG = torch.optim.Adam(G.parameters(), lr_g, betas=(beta_1, beta_2))
optimizerD = torch.optim.Adam(D.parameters(), lr_d, betas=(beta_1, beta_2))

criterion_adv = nn.BCELoss()
criterion_aux = nn.CrossEntropyLoss()

total_step = len(train_loader)

stats = []

for epoch in range(n_epochs):
    for i, (input, target) in enumerate(train_loader):
        images = input.to(device)

        current_batch_size = images.size()[0]

        realLabel = []
        fakeLabel = []

        for i in range(current_batch_size):
            real_smooth = round(random.uniform(0.9, 1.0), 2)
            realLabel.append(real_smooth)

        for i in range(current_batch_size):
            fake_smooth = round(random.uniform(0.0, 0.1), 2)
            fakeLabel.append(fake_smooth)

        realLabel = torch.FloatTensor(realLabel).to(device)
        fakeLabel = torch.FloatTensor(fakeLabel).to(device)

        target = torch.LongTensor(target).to(device)

        ###########
        # TRAIN D #
        ###########

        # on real data
        predictR, predictRLabel = D(images)

        loss_real_adv = criterion_adv(predictR, realLabel)
        loss_real_aux = criterion_aux(predictRLabel, target)

        real_cls_acc = compute_cls_acc(predictRLabel, target)
        real_score = (predictR.sum() / batch_size)

        # on fake data
        latent_value = torch.randn(current_batch_size, latent_size).to(device)

        gen_labels_D = 10 * torch.ones((current_batch_size,), dtype=torch.long, device=device)
        gen_labels_G = torch.LongTensor(np.random.randint(0, n_classes, current_batch_size)).to(device)
        cls_one_hot = torch.zeros(current_batch_size, n_classes, device=device)
        cls_one_hot[torch.arange(current_batch_size), gen_labels_G] = 1.0

        fake_images = G(latent_value, cls_one_hot)

        predictF, predictFLabel = D(fake_images)

        loss_fake_adv = criterion_adv(predictF, fakeLabel)
        loss_fake_aux = criterion_aux(predictFLabel, gen_labels_D)

        lossD = loss_real_adv + loss_fake_adv + loss_real_aux + loss_fake_aux

        optimizerD.zero_grad()
        optimizerG.zero_grad()
        lossD.backward()
        optimizerD.step()

        ###########
        # TRAIN G #
        ###########

        latent_value = torch.randn(current_batch_size, latent_size).to(device)

        gen_labels_G = torch.LongTensor(np.random.randint(0, n_classes, current_batch_size)).to(device)
        cls_one_hot = torch.zeros(current_batch_size, n_classes, device=device)
        cls_one_hot[torch.arange(current_batch_size), gen_labels_G] = 1.0

        fake_images = G(latent_value, cls_one_hot)

        predictF, predictFLabel = D(fake_images)

        lossG_adv = criterion_adv(predictF, realLabel)
        lossG_aux = criterion_aux(predictFLabel, gen_labels_G)

        fake_cls_acc = compute_cls_acc(predictFLabel, gen_labels_G)
        fake_score = (predictF.sum() / batch_size)

        lossG = lossG_adv + lossG_aux

        optimizerD.zero_grad()
        optimizerG.zero_grad()
        lossG.backward()
        optimizerG.step()

    stats_epoch = 'epoch: {}/{} G loss: {:.4f} D loss: {:.4f} fake score: {:.4f} real score: {:.4f} ' \
                  'loss fake cls: {:.4f} loss real cls: {:.4f} fake acc: {:.1f}% real acc: {:.1f}%'.format(
                  epoch + 1, n_epochs, lossG.item(), lossD.item(), fake_score, real_score, loss_fake_aux, loss_real_aux,
                  fake_cls_acc, real_cls_acc)

    stats.append(stats_epoch)

    with open(r'stats.txt', 'w') as file:
        for parameter in stats:
            file.write('{}\n'.format(parameter))

    with torch.no_grad():
        fake = G(fixed_latent, fixed_labels).detach().cpu()
        transform_PIL = transforms.ToPILImage()
        img_list.append(vutils.make_grid(torch.reshape(fake, (100, 3, 64, 64)), nrow=10, normalize=True))
        transform_PIL(img_list[-1]).save('samples/epoch {}.png'.format(epoch + 1))

    if (epoch + 1) % 50 == 0:
        torch.save(G.state_dict(), 'checkpoints/checkpoint epoch {}.pt'.format(epoch + 1))
