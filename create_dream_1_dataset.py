import os
import torch
import torch.nn.functional as F
import csv

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_one_hot = F.one_hot(torch.arange(0, 10), num_classes=len(classes))

list_images = []
list_label = []
list_w = []
list_one_hot = []

directory = os.getcwd()

img_dir = os.path.join(directory, 'samples_no_interpolation_25k')
csv_dir = os.path.join(directory, 'samples_no_interpolation_25k.csv')

# store image names as list
dirListing = os.listdir(img_dir)

for i in dirListing:
    if ".png" in i:
        list_images.append(i)

for i in list_images:
    file = i.split(' ')

    label = file[0]
    list_label.append(label)

    w = 1.0
    list_w.append(w)

# convert class name to class number
for i in range(len(list_label)):
    for j in range(len(classes)):
        if list_label[i] == classes[j]:
            list_label[i] = j


# convert class number to one-hot vector
def weighing(label_one_hot, weight):
    label_one_hot = classes_one_hot[label_one_hot]
    label_one_hot = label_one_hot.tolist()

    for i, value in enumerate(label_one_hot):
        if label_one_hot[i] == 1:
            label_one_hot[i] = 1.0

    return (label_one_hot)


for i in range(len(list_images)):
    label_one_hot = weighing(int(list_label[i]), list_w[i])
    label_one_hot = ', '.join(map(str, label_one_hot))

    list_one_hot.append(label_one_hot)

#  combine image name and one-hot vector:
list_csv = [list(i) for i in zip(list_images, list_one_hot)]

#  save  list as a .csv
with open(csv_dir, 'w') as file:
    write = csv.writer(file)
    write.writerows(list_csv)
