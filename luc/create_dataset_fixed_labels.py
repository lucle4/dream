import os
import torch
import torch.nn.functional as F
import csv

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_one_hot = F.one_hot(torch.arange(0, 10), num_classes=len(classes))

w_1 = 1
w_2 = 1

list_images = []
list_label_1 = []
list_label_2 = []
list_two_hot = []

directory = os.getcwd()

img_dir = os.path.join(directory, 'samples_dreamed_30k')
csv_dir = os.path.join(directory, 'dream_dataset_30k.csv')

# store image names as list
dirListing = os.listdir(img_dir)

for i in dirListing:
     if ".png" in i:
         list_images.append(i)


for i in list_images:

    file = i.split('_')

    label_1 = file[0]
    label_2 = file[2]

    list_label_1.append(label_1)
    list_label_2.append(label_2)


# convert class name to class number
for i in range(len(list_label_1)):
    for j in range(len(classes)):
        if list_label_1[i] == classes[j]:
            list_label_1[i] = j

for i in range(len(list_label_2)):
    for j in range(len(classes)):
        if list_label_2[i] == classes[j]:
            list_label_2[i] = j


# convert class number to one-hot vector
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
    label_combined = ', '. join(map(str, label_combined))

    list_two_hot.append(label_combined)


# combine image name and one-hot vector:
list_csv = [list(i) for i in zip(list_images, list_two_hot)]


# save list as a .csv
with open(csv_dir, 'w') as file:

    write = csv.writer(file)
    write.writerows(list_csv)
