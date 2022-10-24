from __future__ import print_function, division
import os
import torch
import torch.nn.functional as F
import csv

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_one_hot = F.one_hot(torch.arange(0, 10), num_classes=len(classes))

w_1 = 0.7
w_2 = 1 - w_1

list_images = []
list_label_1 = []
list_label_2 = []
list_two_hot = []

img_dir = '/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/create dataset/samples dreamed'
csv_dir = '/Users/luc/Documents/Dokumente/Bildung/Humanmedizin/MA : MD-PhD/Master Thesis/Code/cifar-10 ACGAN/create dataset/dream_dataset.csv'


# store image names in a list
dirListing = os.listdir(img_dir)

for i in dirListing:
     if ".png" in i:
         list_images.append(i)


# convert list items to class name
list_label_1 = [i.split('_')[0] for i in list_images]

list_label_2 = [i.split('7_')[1] for i in list_images]
list_label_2 = [i.split('_0')[0] for i in list_label_2]


# convert class name to class number
for i in range(len(list_label_1)):
    for j in range(len(classes)):
        if list_label_1[i] == classes[j]:
            list_label_1[i] = j

for i in range(len(list_label_2)):
    for j in range(len(classes)):
        if list_label_2[i] == classes[j]:
            list_label_2[i] = j


# convert class number to one-hot label
def weighing(label_one_hot, weight):

    label_one_hot = classes_one_hot[label_one_hot]
    label_one_hot = label_one_hot.tolist()

    for i, value in enumerate(label_one_hot):
        if label_one_hot[i] == 1:
            label_one_hot[i] = round(weight, 2)

    return(label_one_hot)

for i in range(len(list_images)):
    label_1_one_hot = weighing(int(list_label_1[i]), w_1)
    label_2_one_hot = weighing(int(list_label_2[i]), w_2)

    label_combined = [i + j for i, j in zip(label_1_one_hot, label_2_one_hot)]

    list_two_hot.append(label_combined)


# combine image name and one-hot label:
list_csv = [list(i) for i in zip(list_images, list_two_hot)]


# save the list as a .csv
with open(csv_dir, 'w') as file:

    write = csv.writer(file)
    write.writerows(list_csv)
