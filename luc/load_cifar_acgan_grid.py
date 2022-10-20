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

latent_dim = 100
class_dim = 10
gf_dim = 96
df_dim = 32
in_w = in_h = 32
c_dim = 3

w_1 = 0.3
w_2 = 1 - w_1

transform_PIL=transforms.ToPILImage()

def conv_bn_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):

    return nn.Sequential(
		nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),)

def tconv_bn_layer(in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0):

	return nn.Sequential(
		nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding),
		nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),)

def tconv_layer(in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0):

	return nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding)

def conv_layer(in_channels,out_channels,kernel_size,stride=1,padding=0):

	return nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)

def fc_layer(in_features,out_features):

	return nn.Linear(in_features,out_features)

def fc_bn_layer(in_features,out_features):

	return nn.Sequential(
		nn.Linear(in_features,out_features),
		nn.BatchNorm1d(out_features))

def conv_out_size_same(size, stride):

	return int(math.ceil(float(size) / float(stride)))

s_h, s_w = in_h, in_w
s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

class Generator(nn.Module):

	def __init__(self):
		super(Generator,self).__init__()
		self.fc_layer1 = fc_layer(latent_dim+class_dim,gf_dim*8)
		self.up_sample_layer2 = tconv_bn_layer(gf_dim*8,gf_dim*4,4,2,0)
		self.up_sample_layer3 = tconv_bn_layer(gf_dim*4,gf_dim*2,4,2,1)
		self.up_sample_layer4 = tconv_bn_layer(gf_dim*2,gf_dim,4,2,1)
		self.up_sample_layer5 = tconv_layer(gf_dim,c_dim,4,2,1)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = F.relu(self.fc_layer1(x)).view(-1,gf_dim*8,1,1)
		# x = F.relu(self.fc_layer1(x))
		# x = x.view(x.size(0), -1, 1, 1)
		x = F.relu(self.up_sample_layer2(x))
		x = F.relu(self.up_sample_layer3(x))
		x = F.relu(self.up_sample_layer4(x))
		x = self.up_sample_layer5(x)
		return self.tanh(x)



G = Generator().to(device)
G.load_state_dict(torch.load('/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/training 2/checkpoints/checkpoint epoch 90.pt'))
G.eval()

n_images = 10

fixed_latent = torch.randn(100, 100, device=device)
fixed_labels = torch.zeros(100, 10, device=device)

for i in range(10):
     for j in range(10):
         fixed_labels[j*10+i][j] = 0.7

for i in range(10):
    for j in range(10):
        fixed_labels[i*10+j][j] = fixed_labels[i*10+j][j] + 0.3

print(fixed_labels)
print(fixed_labels.size())
fixed_labels = fixed_labels.view(100, 10)

fixed_noise = torch.cat((fixed_latent,fixed_labels), 1)


img_list = []
transform_PIL = transforms.ToPILImage()

with torch.no_grad():
    img = G(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(torch.reshape(img,(100,c_dim,in_h,in_w)),nrow=10, padding=2, normalize=True))
    transform_PIL(img_list[-1]).save('/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/training 2/samples dreamed/grid.png')
