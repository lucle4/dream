import os
import numpy as np
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from torchvision.io import read_image
import torchvision.models as models
import torch.nn as nn
from torchvision.models import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

checkpoint_dir = '/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ResNet/original + dreamed/0k : 50k/higher label, random weighing/checkpoint epoch 100.pt'
label_dir = '/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ResNet/datasets/noise dataset/evaluation_dataset.csv'
img_dir = '/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ResNet/datasets/noise dataset/snp/snp (0.32)'

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ConvertImageDtype(float),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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


dataset = Dataset(label_dir, img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


model = resnet50(weights=None)
model.fc = nn.Linear(2048, 10)

if torch.cuda.is_available():
    model.cuda()

#model.load_state_dict(torch.load(os.path.join(directory, checkpoint_dir)))
model.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device('cpu')))
model.eval()


accuracy_plane = 0.0
accuracy_car = 0.0
accuracy_bird = 0.0
accuracy_cat = 0.0
accuracy_deer = 0.0
accuracy_dog = 0.0
accuracy_frog = 0.0
accuracy_horse = 0.0
accuracy_ship = 0.0
accuracy_truck = 0.0

n_plane = 0
n_car = 0
n_bird = 0
n_cat = 0
n_deer = 0
n_dog = 0
n_frog = 0
n_horse = 0
n_ship = 0
n_truck = 0

plane_plane = []
plane_car = []
plane_bird = []
plane_cat = []
plane_deer = []
plane_dog = []
plane_frog = []
plane_horse = []
plane_ship = []
plane_truck = []

car_plane = []
car_car = []
car_bird = []
car_cat = []
car_deer = []
car_dog = []
car_frog = []
car_horse = []
car_ship = []
car_truck = []

bird_plane = []
bird_car = []
bird_bird = []
bird_cat = []
bird_deer = []
bird_dog = []
bird_frog = []
bird_horse = []
bird_ship = []
bird_truck = []

cat_plane = []
cat_car = []
cat_bird = []
cat_cat = []
cat_deer = []
cat_dog = []
cat_frog = []
cat_horse = []
cat_ship = []
cat_truck = []

deer_plane = []
deer_car = []
deer_bird = []
deer_cat = []
deer_deer = []
deer_dog = []
deer_frog = []
deer_horse = []
deer_ship = []
deer_truck = []

dog_plane = []
dog_car = []
dog_bird = []
dog_cat = []
dog_deer = []
dog_dog = []
dog_frog = []
dog_horse = []
dog_ship = []
dog_truck = []

frog_plane = []
frog_car = []
frog_bird = []
frog_cat = []
frog_deer = []
frog_dog = []
frog_frog = []
frog_horse = []
frog_ship = []
frog_truck = []

horse_plane = []
horse_car = []
horse_bird = []
horse_cat = []
horse_deer = []
horse_dog = []
horse_frog = []
horse_horse = []
horse_ship = []
horse_truck = []

ship_plane = []
ship_car = []
ship_bird = []
ship_cat = []
ship_deer = []
ship_dog = []
ship_frog = []
ship_horse = []
ship_ship = []
ship_truck = []

truck_plane = []
truck_car = []
truck_bird = []
truck_cat = []
truck_deer = []
truck_dog = []
truck_frog = []
truck_horse = []
truck_ship = []
truck_truck = []


for i, (image, label) in enumerate(dataloader):

    image = image.to(device)
    label = label.to(device)
    label = torch.argmax(label, 1)

    output = model(image.float())

    _, predicted = torch.max(output, 1)
    output = output.view(10)
    output = output.tolist()

    if label.item() == 0:
        n_plane += 1
        accuracy_plane += (predicted == label).sum().item()

        plane_plane.append(output[0])
        plane_car.append(output[1])
        plane_bird.append(output[2])
        plane_cat.append(output[3])
        plane_deer.append(output[4])
        plane_dog.append(output[5])
        plane_frog.append(output[6])
        plane_horse.append(output[7])
        plane_ship.append(output[8])
        plane_truck.append(output[9])


    elif label.item() == 1:
        n_car += 1
        accuracy_car += (predicted == label).sum().item()

        car_plane.append(output[0])
        car_car.append(output[1])
        car_bird.append(output[2])
        car_cat.append(output[3])
        car_deer.append(output[4])
        car_dog.append(output[5])
        car_frog.append(output[6])
        car_horse.append(output[7])
        car_ship.append(output[8])
        car_truck.append(output[9])


    elif label.item() == 2:
        n_bird += 1
        accuracy_bird += (predicted == label).sum().item()

        bird_plane.append(output[0])
        bird_car.append(output[1])
        bird_bird.append(output[2])
        bird_cat.append(output[3])
        bird_deer.append(output[4])
        bird_dog.append(output[5])
        bird_frog.append(output[6])
        bird_horse.append(output[7])
        bird_ship.append(output[8])
        bird_truck.append(output[9])


    elif label.item() == 3:
        n_cat += 1
        accuracy_cat += (predicted == label).sum().item()

        cat_plane.append(output[0])
        cat_car.append(output[1])
        cat_bird.append(output[2])
        cat_cat.append(output[3])
        cat_deer.append(output[4])
        cat_dog.append(output[5])
        cat_frog.append(output[6])
        cat_horse.append(output[7])
        cat_ship.append(output[8])
        cat_truck.append(output[9])


    elif label.item() == 4:
        n_deer += 1
        accuracy_deer += (predicted == label).sum().item()

        deer_plane.append(output[0])
        deer_car.append(output[1])
        deer_bird.append(output[2])
        deer_cat.append(output[3])
        deer_deer.append(output[4])
        deer_dog.append(output[5])
        deer_frog.append(output[6])
        deer_horse.append(output[7])
        deer_ship.append(output[8])
        deer_truck.append(output[9])


    elif label.item() == 5:
        n_dog += 1
        accuracy_dog += (predicted == label).sum().item()

        dog_plane.append(output[0])
        dog_car.append(output[1])
        dog_bird.append(output[2])
        dog_cat.append(output[3])
        dog_deer.append(output[4])
        dog_dog.append(output[5])
        dog_frog.append(output[6])
        dog_horse.append(output[7])
        dog_ship.append(output[8])
        dog_truck.append(output[9])


    elif label.item() == 6:
        n_frog += 1
        accuracy_frog += (predicted == label).sum().item()

        frog_plane.append(output[0])
        frog_car.append(output[1])
        frog_bird.append(output[2])
        frog_cat.append(output[3])
        frog_deer.append(output[4])
        frog_dog.append(output[5])
        frog_frog.append(output[6])
        frog_horse.append(output[7])
        frog_ship.append(output[8])
        frog_truck.append(output[9])


    elif label.item() == 7:
        n_horse += 1
        accuracy_horse += (predicted == label).sum().item()

        horse_plane.append(output[0])
        horse_car.append(output[1])
        horse_bird.append(output[2])
        horse_cat.append(output[3])
        horse_deer.append(output[4])
        horse_dog.append(output[5])
        horse_frog.append(output[6])
        horse_horse.append(output[7])
        horse_ship.append(output[8])
        horse_truck.append(output[9])


    elif label.item() == 8:
        n_ship += 1
        accuracy_ship += (predicted == label).sum().item()
        ship_plane.append(output[0])
        ship_car.append(output[1])
        ship_bird.append(output[2])
        ship_cat.append(output[3])
        ship_deer.append(output[4])
        ship_dog.append(output[5])
        ship_frog.append(output[6])
        ship_horse.append(output[7])
        ship_ship.append(output[8])
        ship_truck.append(output[9])

    elif label.item() == 9:
        n_truck += 1
        accuracy_truck += (predicted == label).sum().item()

        truck_plane.append(output[0])
        truck_car.append(output[1])
        truck_bird.append(output[2])
        truck_cat.append(output[3])
        truck_deer.append(output[4])
        truck_dog.append(output[5])
        truck_frog.append(output[6])
        truck_horse.append(output[7])
        truck_ship.append(output[8])
        truck_truck.append(output[9])


accuracy_plane =  (accuracy_plane / n_plane) * 100
accuracy_car =  (accuracy_car / n_car) * 100
accuracy_bird =  (accuracy_bird / n_bird) * 100
accuracy_cat =  (accuracy_cat / n_cat) * 100
accuracy_deer =  (accuracy_deer / n_deer) * 100
accuracy_dog =  (accuracy_dog / n_dog) * 100
accuracy_frog =  (accuracy_frog / n_frog) * 100
accuracy_horse =  (accuracy_horse / n_horse) * 100
accuracy_ship =  (accuracy_ship / n_ship) * 100
accuracy_truck =  (accuracy_truck / n_truck) * 100


dict_performance = {'accuracy plane  ': accuracy_plane, 'accuracy car    ': accuracy_car, 'accuracy bird   ': accuracy_bird,
                    'accuracy cat    ': accuracy_cat, 'accuracy deer   ': accuracy_deer, 'accuracy dog    ': accuracy_dog,
                    'accuracy frog   ': accuracy_frog, 'accuracy horse  ': accuracy_horse, 'accuracy ship   ': accuracy_ship,
                    'accuracy truck  ': accuracy_truck}

dict_performance_sorted = sorted(dict_performance.items(), key=lambda x:x[1])


checkpoint = checkpoint_dir.split('cifar-10 ResNet')
images = img_dir.split('cifar-10 ResNet')

print('')
print('network:        ', checkpoint[1])
print('test images:    ', images[1])
print('')

for (key, value) in dict_performance_sorted:
    print('{} {:.2f}%'.format(key, value))

print('')
print('')

print('PLANE    SD      MEAN')
print('plane:   {:.3f}   {:.3f}'.format(np.std(plane_plane), np.mean(plane_plane)))
print('car:     {:.3f}   {:.3f}'.format(np.std(plane_car), np.mean(plane_car)))
print('bird:    {:.3f}   {:.3f}'.format(np.std(plane_bird), np.mean(plane_bird)))
print('cat:     {:.3f}   {:.3f}'.format(np.std(plane_cat), np.mean(plane_cat)))
print('deer:    {:.3f}   {:.3f}'.format(np.std(plane_deer), np.mean(plane_deer)))
print('dog:     {:.3f}   {:.3f}'.format(np.std(plane_dog), np.mean(plane_dog)))
print('frog:    {:.3f}   {:.3f}'.format(np.std(plane_frog), np.mean(plane_frog)))
print('horse:   {:.3f}   {:.3f}'.format(np.std(plane_horse), np.mean(plane_horse)))
print('ship:    {:.3f}   {:.3f}'.format(np.std(plane_ship), np.mean(plane_ship)))
print('truck:   {:.3f}   {:.3f}'.format(np.std(plane_truck), np.mean(plane_truck)))
print('')

print('CAR      SD      MEAN')
print('plane:   {:.3f}   {:.3f}'.format(np.std(car_plane), np.mean(car_plane)))
print('car:     {:.3f}   {:.3f}'.format(np.std(car_car), np.mean(car_car)))
print('bird:    {:.3f}   {:.3f}'.format(np.std(car_bird), np.mean(car_bird)))
print('cat:     {:.3f}   {:.3f}'.format(np.std(car_cat), np.mean(car_cat)))
print('deer:    {:.3f}   {:.3f}'.format(np.std(car_deer), np.mean(car_deer)))
print('dog:     {:.3f}   {:.3f}'.format(np.std(car_dog), np.mean(car_dog)))
print('frog:    {:.3f}   {:.3f}'.format(np.std(car_frog), np.mean(car_frog)))
print('horse:   {:.3f}   {:.3f}'.format(np.std(car_horse), np.mean(car_horse)))
print('ship:    {:.3f}   {:.3f}'.format(np.std(car_ship), np.mean(car_ship)))
print('truck:   {:.3f}   {:.3f}'.format(np.std(car_truck), np.mean(car_truck)))
print('')

print('BIRD     SD      MEAN')
print('plane:   {:.3f}   {:.3f}'.format(np.std(bird_plane), np.mean(bird_plane)))
print('car:     {:.3f}   {:.3f}'.format(np.std(bird_car), np.mean(bird_car)))
print('bird:    {:.3f}   {:.3f}'.format(np.std(bird_bird), np.mean(bird_bird)))
print('cat:     {:.3f}   {:.3f}'.format(np.std(bird_cat), np.mean(bird_cat)))
print('deer:    {:.3f}   {:.3f}'.format(np.std(bird_deer), np.mean(bird_deer)))
print('dog:     {:.3f}   {:.3f}'.format(np.std(bird_dog), np.mean(bird_dog)))
print('frog:    {:.3f}   {:.3f}'.format(np.std(bird_frog), np.mean(bird_frog)))
print('horse:   {:.3f}   {:.3f}'.format(np.std(bird_horse), np.mean(bird_horse)))
print('ship:    {:.3f}   {:.3f}'.format(np.std(bird_ship), np.mean(bird_ship)))
print('truck:   {:.3f}   {:.3f}'.format(np.std(bird_truck), np.mean(bird_truck)))
print('')

print('CAT      SD      MEAN')
print('plane:   {:.3f}   {:.3f}'.format(np.std(cat_plane), np.mean(cat_plane)))
print('car:     {:.3f}   {:.3f}'.format(np.std(cat_car), np.mean(cat_car)))
print('bird:    {:.3f}   {:.3f}'.format(np.std(cat_bird), np.mean(cat_bird)))
print('cat:     {:.3f}   {:.3f}'.format(np.std(cat_cat), np.mean(cat_cat)))
print('deer:    {:.3f}   {:.3f}'.format(np.std(cat_deer), np.mean(cat_deer)))
print('dog:     {:.3f}   {:.3f}'.format(np.std(cat_dog), np.mean(cat_dog)))
print('frog:    {:.3f}   {:.3f}'.format(np.std(cat_frog), np.mean(cat_frog)))
print('horse:   {:.3f}   {:.3f}'.format(np.std(cat_horse), np.mean(cat_horse)))
print('ship:    {:.3f}   {:.3f}'.format(np.std(cat_ship), np.mean(cat_ship)))
print('truck:   {:.3f}   {:.3f}'.format(np.std(cat_truck), np.mean(cat_truck)))
print('')

print('DEER     SD      MEAN')
print('plane:   {:.3f}   {:.3f}'.format(np.std(deer_plane), np.mean(deer_plane)))
print('car:     {:.3f}   {:.3f}'.format(np.std(deer_car), np.mean(deer_car)))
print('bird:    {:.3f}   {:.3f}'.format(np.std(deer_bird), np.mean(deer_bird)))
print('cat:     {:.3f}   {:.3f}'.format(np.std(deer_cat), np.mean(deer_cat)))
print('deer:    {:.3f}   {:.3f}'.format(np.std(deer_deer), np.mean(deer_deer)))
print('dog:     {:.3f}   {:.3f}'.format(np.std(deer_dog), np.mean(deer_dog)))
print('frog:    {:.3f}   {:.3f}'.format(np.std(deer_frog), np.mean(deer_frog)))
print('horse:   {:.3f}   {:.3f}'.format(np.std(deer_horse), np.mean(deer_horse)))
print('ship:    {:.3f}   {:.3f}'.format(np.std(deer_ship), np.mean(deer_ship)))
print('truck:   {:.3f}   {:.3f}'.format(np.std(deer_truck), np.mean(deer_truck)))
print('')

print('DOG      SD      MEAN')
print('plane:   {:.3f}   {:.3f}'.format(np.std(dog_plane), np.mean(dog_plane)))
print('car:     {:.3f}   {:.3f}'.format(np.std(dog_car), np.mean(dog_car)))
print('bird:    {:.3f}   {:.3f}'.format(np.std(dog_bird), np.mean(dog_bird)))
print('cat:     {:.3f}   {:.3f}'.format(np.std(dog_cat), np.mean(dog_cat)))
print('deer:    {:.3f}   {:.3f}'.format(np.std(dog_deer), np.mean(dog_deer)))
print('dog:     {:.3f}   {:.3f}'.format(np.std(dog_dog), np.mean(dog_dog)))
print('frog:    {:.3f}   {:.3f}'.format(np.std(dog_frog), np.mean(dog_frog)))
print('horse:   {:.3f}   {:.3f}'.format(np.std(dog_horse), np.mean(dog_horse)))
print('ship:    {:.3f}   {:.3f}'.format(np.std(dog_ship), np.mean(dog_ship)))
print('truck:   {:.3f}   {:.3f}'.format(np.std(dog_truck), np.mean(dog_truck)))
print('')

print('FROG     SD      MEAN')
print('plane:   {:.3f}   {:.3f}'.format(np.std(frog_plane), np.mean(frog_plane)))
print('car:     {:.3f}   {:.3f}'.format(np.std(frog_car), np.mean(frog_car)))
print('bird:    {:.3f}   {:.3f}'.format(np.std(frog_bird), np.mean(frog_bird)))
print('cat:     {:.3f}   {:.3f}'.format(np.std(frog_cat), np.mean(frog_cat)))
print('deer:    {:.3f}   {:.3f}'.format(np.std(frog_deer), np.mean(frog_deer)))
print('dog:     {:.3f}   {:.3f}'.format(np.std(frog_dog), np.mean(frog_dog)))
print('frog:    {:.3f}   {:.3f}'.format(np.std(frog_frog), np.mean(frog_frog)))
print('horse:   {:.3f}   {:.3f}'.format(np.std(frog_horse), np.mean(frog_horse)))
print('ship:    {:.3f}   {:.3f}'.format(np.std(frog_ship), np.mean(frog_ship)))
print('truck:   {:.3f}   {:.3f}'.format(np.std(frog_truck), np.mean(frog_truck)))
print('')

print('HORSE    SD      MEAN')
print('plane:   {:.3f}   {:.3f}'.format(np.std(horse_plane), np.mean(horse_plane)))
print('car:     {:.3f}   {:.3f}'.format(np.std(horse_car), np.mean(horse_car)))
print('bird:    {:.3f}   {:.3f}'.format(np.std(horse_bird), np.mean(horse_bird)))
print('cat:     {:.3f}   {:.3f}'.format(np.std(horse_cat), np.mean(horse_cat)))
print('deer:    {:.3f}   {:.3f}'.format(np.std(horse_deer), np.mean(horse_deer)))
print('dog:     {:.3f}   {:.3f}'.format(np.std(horse_dog), np.mean(horse_dog)))
print('frog:    {:.3f}   {:.3f}'.format(np.std(horse_frog), np.mean(horse_frog)))
print('horse:   {:.3f}   {:.3f}'.format(np.std(horse_horse), np.mean(horse_horse)))
print('ship:    {:.3f}   {:.3f}'.format(np.std(horse_ship), np.mean(horse_ship)))
print('truck:   {:.3f}   {:.3f}'.format(np.std(horse_truck), np.mean(horse_truck)))
print('')

print('SHIP     SD      MEAN')
print('plane:   {:.3f}   {:.3f}'.format(np.std(ship_plane), np.mean(ship_plane)))
print('car:     {:.3f}   {:.3f}'.format(np.std(ship_car), np.mean(ship_car)))
print('bird:    {:.3f}   {:.3f}'.format(np.std(ship_bird), np.mean(ship_bird)))
print('cat:     {:.3f}   {:.3f}'.format(np.std(ship_cat), np.mean(ship_cat)))
print('deer:    {:.3f}   {:.3f}'.format(np.std(ship_deer), np.mean(ship_deer)))
print('dog:     {:.3f}   {:.3f}'.format(np.std(ship_dog), np.mean(ship_dog)))
print('frog:    {:.3f}   {:.3f}'.format(np.std(ship_frog), np.mean(ship_frog)))
print('horse:   {:.3f}   {:.3f}'.format(np.std(ship_horse), np.mean(ship_horse)))
print('ship:    {:.3f}   {:.3f}'.format(np.std(ship_ship), np.mean(ship_ship)))
print('truck:   {:.3f}   {:.3f}'.format(np.std(ship_truck), np.mean(ship_truck)))
print('')

print('TRUCK    SD      MEAN')
print('plane:   {:.3f}   {:.3f}'.format(np.std(truck_plane), np.mean(truck_plane)))
print('car:     {:.3f}   {:.3f}'.format(np.std(truck_car), np.mean(truck_car)))
print('bird:    {:.3f}   {:.3f}'.format(np.std(truck_bird), np.mean(truck_bird)))
print('cat:     {:.3f}   {:.3f}'.format(np.std(truck_cat), np.mean(truck_cat)))
print('deer:    {:.3f}   {:.3f}'.format(np.std(truck_deer), np.mean(truck_deer)))
print('dog:     {:.3f}   {:.3f}'.format(np.std(truck_dog), np.mean(truck_dog)))
print('frog:    {:.3f}   {:.3f}'.format(np.std(truck_frog), np.mean(truck_frog)))
print('horse:   {:.3f}   {:.3f}'.format(np.std(truck_horse), np.mean(truck_horse)))
print('ship:    {:.3f}   {:.3f}'.format(np.std(truck_ship), np.mean(truck_ship)))
print('truck:   {:.3f}   {:.3f}'.format(np.std(truck_truck), np.mean(truck_truck)))
