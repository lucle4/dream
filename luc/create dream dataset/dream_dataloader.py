import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

img_dir = '/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/create dataset/samples dreamed'
label_dir = '/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/create dataset/dream_dataset.csv'

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ConvertImageDtype(float),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class DreamDataset(Dataset):

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
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

dream_dataset = DreamDataset(label_dir, img_dir, transform=transform)
train_loader = DataLoader(dream_dataset, batch_size=10, shuffle=True)

dream_images, dream_labels = next(iter(train_loader))


# save the first sample
img = dream_images[0]
print(dream_labels[0])

save_image(img, './dream_dataloader_test.png', padding=2, normalize=True)
