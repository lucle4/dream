import os
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import torch.nn as nn
from torchvision.models import resnet50
from csv import writer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

loaded_trial = 3

directory = os.getcwd()


label_dir = os.path.join(directory, 'datasets/noise dataset/evaluation_dataset.csv')

img_dir_original = os.path.join(directory, 'datasets/noise dataset/original')

img_dir_noise_1_1 = os.path.join(directory, 'datasets/noise dataset/gaussian/gaussian (0.04)')
img_dir_noise_1_2 = os.path.join(directory, 'datasets/noise dataset/gaussian/gaussian (0.08)')
img_dir_noise_1_3 = os.path.join(directory, 'datasets/noise dataset/gaussian/gaussian (0.16)')
img_dir_noise_1_4 = os.path.join(directory, 'datasets/noise dataset/gaussian/gaussian (0.32)')
img_dir_noise_1_5 = os.path.join(directory, 'datasets/noise dataset/gaussian/gaussian (0.64)')
img_dir_noise_1_6 = os.path.join(directory, 'datasets/noise dataset/gaussian/gaussian (1.28)')

img_dir_noise_2_1 = os.path.join(directory, 'datasets/noise dataset/speckle/speckle (0.16)')
img_dir_noise_2_2 = os.path.join(directory, 'datasets/noise dataset/speckle/speckle (0.32)')
img_dir_noise_2_3 = os.path.join(directory, 'datasets/noise dataset/speckle/speckle (0.64)')
img_dir_noise_2_4 = os.path.join(directory, 'datasets/noise dataset/speckle/speckle (1.28)')
img_dir_noise_2_5 = os.path.join(directory, 'datasets/noise dataset/speckle/speckle (2.56)')
img_dir_noise_2_6 = os.path.join(directory, 'datasets/noise dataset/speckle/speckle (5.12)')

img_dir_noise_3_1 = os.path.join(directory, 'datasets/noise dataset/snp/snp (0.02)')
img_dir_noise_3_2 = os.path.join(directory, 'datasets/noise dataset/snp/snp (0.04)')
img_dir_noise_3_3 = os.path.join(directory, 'datasets/noise dataset/snp/snp (0.08)')
img_dir_noise_3_4 = os.path.join(directory, 'datasets/noise dataset/snp/snp (0.16)')
img_dir_noise_3_5 = os.path.join(directory, 'datasets/noise dataset/snp/snp (0.32)')
img_dir_noise_3_6 = os.path.join(directory, 'datasets/noise dataset/snp/snp (0.64)')

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ConvertImageDtype(float),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class Dataset(Dataset):
    def __init__(self, label_dir, img_dir, transform=None, target_transform=None):
        self.img_label = pd.read_csv(label_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_label.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_label.iloc[idx, 1]

        label = [float(i) for i in label.split(',')]
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


model_50k_0k = resnet50(weights=None)
model_50k_0k.fc = nn.Linear(2048, 10)

model_40k_10k = resnet50(weights=None)
model_40k_10k.fc = nn.Linear(2048, 10)

model_30k_20k = resnet50(weights=None)
model_30k_20k.fc = nn.Linear(2048, 10)

model_20k_30k = resnet50(weights=None)
model_20k_30k.fc = nn.Linear(2048, 10)

model_10k_40k = resnet50(weights=None)
model_10k_40k.fc = nn.Linear(2048, 10)

model_0k_50k_1 = resnet50(weights=None)
model_0k_50k_1.fc = nn.Linear(2048, 10)

model_0k_50k_2 = resnet50(weights=None)
model_0k_50k_2.fc = nn.Linear(2048, 10)

if torch.cuda.is_available():
    model_50k_0k.cuda()
    model_40k_10k.cuda()
    model_30k_20k.cuda()
    model_20k_30k.cuda()
    model_10k_40k.cuda()
    model_0k_50k_1.cuda()
    model_0k_50k_2.cuda()

model_50k_0k.load_state_dict(torch.load(
    os.path.join(directory, 'final training mix/trial {}/50k_0k/checkpoint epoch 100.pt'.format(loaded_trial)),
    map_location=torch.device('cpu')))
model_50k_0k.eval()

model_40k_10k.load_state_dict(torch.load(
    os.path.join(directory, 'final training mix/trial {}/40k_10k/checkpoint epoch 100.pt'.format(loaded_trial)),
    map_location=torch.device('cpu')))
model_40k_10k.eval()

model_30k_20k.load_state_dict(torch.load(
    os.path.join(directory, 'final training mix/trial {}/30k_20k/checkpoint epoch 100.pt'.format(loaded_trial)),
    map_location=torch.device('cpu')))
model_30k_20k.eval()

model_20k_30k.load_state_dict(torch.load(
    os.path.join(directory, 'final training mix/trial {}/20k_30k/checkpoint epoch 100.pt'.format(loaded_trial)),
    map_location=torch.device('cpu')))
model_20k_30k.eval()

model_10k_40k.load_state_dict(torch.load(
    os.path.join(directory, 'final training mix/trial {}/10k_40k/checkpoint epoch 100.pt'.format(loaded_trial)),
    map_location=torch.device('cpu')))
model_10k_40k.eval()

model_0k_50k_1.load_state_dict(torch.load(
    os.path.join(directory, 'final training mix/trial {}/0k_50k_1/checkpoint epoch 100.pt'.format(loaded_trial)),
    map_location=torch.device('cpu')))
model_0k_50k_1.eval()

model_0k_50k_2.load_state_dict(torch.load(
    os.path.join(directory, 'final training mix/trial {}/0k_50k_2/checkpoint epoch 100.pt'.format(loaded_trial)),
    map_location=torch.device('cpu')))
model_0k_50k_2.eval()

original_dataset = Dataset(label_dir, img_dir_original, transform=transform)
original_loader = DataLoader(original_dataset, batch_size=1, shuffle=False)


noise_1_1_dataset = Dataset(label_dir, img_dir_noise_1_1, transform=transform)
noise_1_2_dataset = Dataset(label_dir, img_dir_noise_1_2, transform=transform)
noise_1_3_dataset = Dataset(label_dir, img_dir_noise_1_3, transform=transform)
noise_1_4_dataset = Dataset(label_dir, img_dir_noise_1_4, transform=transform)
noise_1_5_dataset = Dataset(label_dir, img_dir_noise_1_5, transform=transform)
noise_1_6_dataset = Dataset(label_dir, img_dir_noise_1_6, transform=transform)

noise_1_1_loader = DataLoader(noise_1_1_dataset, batch_size=1, shuffle=False)
noise_1_2_loader = DataLoader(noise_1_2_dataset, batch_size=1, shuffle=False)
noise_1_3_loader = DataLoader(noise_1_3_dataset, batch_size=1, shuffle=False)
noise_1_4_loader = DataLoader(noise_1_4_dataset, batch_size=1, shuffle=False)
noise_1_5_loader = DataLoader(noise_1_5_dataset, batch_size=1, shuffle=False)
noise_1_6_loader = DataLoader(noise_1_6_dataset, batch_size=1, shuffle=False)


noise_2_1_dataset = Dataset(label_dir, img_dir_noise_2_1, transform=transform)
noise_2_2_dataset = Dataset(label_dir, img_dir_noise_2_2, transform=transform)
noise_2_3_dataset = Dataset(label_dir, img_dir_noise_2_3, transform=transform)
noise_2_4_dataset = Dataset(label_dir, img_dir_noise_2_4, transform=transform)
noise_2_5_dataset = Dataset(label_dir, img_dir_noise_2_5, transform=transform)
noise_2_6_dataset = Dataset(label_dir, img_dir_noise_2_6, transform=transform)

noise_2_1_loader = DataLoader(noise_2_1_dataset, batch_size=1, shuffle=False)
noise_2_2_loader = DataLoader(noise_2_2_dataset, batch_size=1, shuffle=False)
noise_2_3_loader = DataLoader(noise_2_3_dataset, batch_size=1, shuffle=False)
noise_2_4_loader = DataLoader(noise_2_4_dataset, batch_size=1, shuffle=False)
noise_2_5_loader = DataLoader(noise_2_5_dataset, batch_size=1, shuffle=False)
noise_2_6_loader = DataLoader(noise_2_6_dataset, batch_size=1, shuffle=False)


noise_3_1_dataset = Dataset(label_dir, img_dir_noise_3_1, transform=transform)
noise_3_2_dataset = Dataset(label_dir, img_dir_noise_3_2, transform=transform)
noise_3_3_dataset = Dataset(label_dir, img_dir_noise_3_3, transform=transform)
noise_3_4_dataset = Dataset(label_dir, img_dir_noise_3_4, transform=transform)
noise_3_5_dataset = Dataset(label_dir, img_dir_noise_3_5, transform=transform)
noise_3_6_dataset = Dataset(label_dir, img_dir_noise_3_6, transform=transform)

noise_3_1_loader = DataLoader(noise_3_1_dataset, batch_size=1, shuffle=False)
noise_3_2_loader = DataLoader(noise_3_2_dataset, batch_size=1, shuffle=False)
noise_3_3_loader = DataLoader(noise_3_3_dataset, batch_size=1, shuffle=False)
noise_3_4_loader = DataLoader(noise_3_4_dataset, batch_size=1, shuffle=False)
noise_3_5_loader = DataLoader(noise_3_5_dataset, batch_size=1, shuffle=False)
noise_3_6_loader = DataLoader(noise_3_6_dataset, batch_size=1, shuffle=False)

model_list = ['50k original, 0 dream', '40k original, 10k dream 1', '30k original, 20k dream 1',
              '20k original, 30k dream 1', '10k original, 40k dream 1', '0 original, 50k dream 1',
              '0 original, 50k dream 2']

all_accuracies = []

n_label = [0 for i in range(10)]

softmax = nn.Softmax(dim=0)


def make_matrix(model_output, current_label, matrix_n):
    for i in range(len(classes)):
        if i == current_label.item():
            n_label[i] += 1

    for j in range(len(classes)):
        if j == current_label:
            if n_label[j] == 1:
                matrix_n[j] = model_output
            else:
                for k in range(len(classes)):
                    matrix_n[j][k] = (((matrix_n[j][k] * (n_label[j] - 1)) + model_output[k]) / n_label[j])

    return matrix_n


def evaluate(dataloader):
    matrix_1 = [[0 for i in range(10)] for i in range(10)]
    matrix_2 = [[0 for i in range(10)] for i in range(10)]
    matrix_3 = [[0 for i in range(10)] for i in range(10)]
    matrix_4 = [[0 for i in range(10)] for i in range(10)]
    matrix_5 = [[0 for i in range(10)] for i in range(10)]
    matrix_6 = [[0 for i in range(10)] for i in range(10)]
    matrix_7 = [[0 for i in range(10)] for i in range(10)]

    running_accuracy_50k_0k = 0
    running_accuracy_40k_10k = 0
    running_accuracy_30k_20k = 0
    running_accuracy_20k_30k = 0
    running_accuracy_10k_40k = 0
    running_accuracy_0k_50k_1 = 0
    running_accuracy_0k_50k_2 = 0

    global n_label
    n_label = [0 for i in range(10)]
    total = 0

    with torch.no_grad():
        for i, (image, label) in enumerate(dataloader):
            print(i)
            if i == 50:
                break

            total += 1

            image = image.to(device)
            label = label.to(device)
            label_idx = torch.argmax(label, 1)

            output_50k_0k = model_50k_0k(image.float())
            _, predicted = torch.max(output_50k_0k, 1)
            running_accuracy_50k_0k += (predicted == label_idx).sum().item()
            output_50k_0k = output_50k_0k.view(10)
            # output_50k_0k = softmax(output_50k_0k)
            output_50k_0k = output_50k_0k.tolist()
            matrix_50k_0k_original = make_matrix(output_50k_0k, label_idx, matrix_1)

            output_40k_10k = model_40k_10k(image.float())
            _, predicted = torch.max(output_40k_10k, 1)
            running_accuracy_40k_10k += (predicted == label_idx).sum().item()
            output_40k_10k = output_40k_10k.view(10)
            # output_40k_10k = softmax(output_40k_10k)
            output_40k_10k = output_40k_10k.tolist()
            matrix_40k_10k_original = make_matrix(output_40k_10k, label_idx, matrix_2)

            output_30k_20k = model_30k_20k(image.float())
            _, predicted = torch.max(output_30k_20k, 1)
            running_accuracy_30k_20k += (predicted == label_idx).sum().item()
            output_30k_20k = output_30k_20k.view(10)
            # output_30k_20k = softmax(output_30k_20k)
            output_30k_20k = output_30k_20k.tolist()
            matrix_30k_20k_original = make_matrix(output_30k_20k, label_idx, matrix_3)

            output_20k_30k = model_20k_30k(image.float())
            _, predicted = torch.max(output_20k_30k, 1)
            running_accuracy_20k_30k += (predicted == label_idx).sum().item()
            output_20k_30k = output_20k_30k.view(10)
            # output_20k_30k = softmax(output_20k_30k)
            output_20k_30k = output_20k_30k.tolist()
            matrix_20k_30k_original = make_matrix(output_20k_30k, label_idx, matrix_4)

            output_10k_40k = model_10k_40k(image.float())
            _, predicted = torch.max(output_10k_40k, 1)
            running_accuracy_10k_40k += (predicted == label_idx).sum().item()
            output_10k_40k = output_10k_40k.view(10)
            # output_10k_40k = softmax(output_10k_40k)
            output_10k_40k = output_10k_40k.tolist()
            matrix_10k_40k_original = make_matrix(output_10k_40k, label_idx, matrix_5)

            output_0k_50k_1 = model_0k_50k_1(image.float())
            _, predicted = torch.max(output_0k_50k_1, 1)
            running_accuracy_0k_50k_1 += (predicted == label_idx).sum().item()
            output_0k_50k_1 = output_0k_50k_1.view(10)
            # output_0k_50k_1 = softmax(output_0k_50k_1)
            output_0k_50k_1 = output_0k_50k_1.tolist()
            matrix_0k_50k_1_original = make_matrix(output_0k_50k_1, label_idx, matrix_6)

            output_0k_50k_2 = model_0k_50k_2(image.float())
            _, predicted = torch.max(output_0k_50k_2, 1)
            running_accuracy_0k_50k_2 += (predicted == label_idx).sum().item()
            output_0k_50k_2 = output_0k_50k_2.view(10)
            # output_0k_50k_2 = softmax(output_0k_50k_2)
            output_0k_50k_2 = output_0k_50k_2.tolist()
            matrix_0k_50k_2_original = make_matrix(output_0k_50k_2, label_idx, matrix_7)

    for i in range(len(classes)):
        for j in range(len(classes)):
            matrix_50k_0k_original[i][j] = float('{:.2f}'.format(matrix_50k_0k_original[i][j]))
            matrix_40k_10k_original[i][j] = float('{:.2f}'.format(matrix_40k_10k_original[i][j]))
            matrix_30k_20k_original[i][j] = float('{:.2f}'.format(matrix_30k_20k_original[i][j]))
            matrix_20k_30k_original[i][j] = float('{:.2f}'.format(matrix_20k_30k_original[i][j]))
            matrix_10k_40k_original[i][j] = float('{:.2f}'.format(matrix_10k_40k_original[i][j]))
            matrix_0k_50k_1_original[i][j] = float('{:.2f}'.format(matrix_0k_50k_1_original[i][j]))
            matrix_0k_50k_2_original[i][j] = float('{:.2f}'.format(matrix_0k_50k_2_original[i][j]))

    matrices = [matrix_50k_0k_original, matrix_40k_10k_original, matrix_30k_20k_original,
                matrix_20k_30k_original, matrix_10k_40k_original, matrix_0k_50k_1_original,
                matrix_0k_50k_2_original]

    accuracy_50k_0k = running_accuracy_50k_0k / total
    accuracy_40k_10k = running_accuracy_40k_10k / total
    accuracy_30k_20k = running_accuracy_30k_20k / total
    accuracy_20k_30k = running_accuracy_20k_30k / total
    accuracy_10k_40k = running_accuracy_10k_40k / total
    accuracy_0k_50k_1 = running_accuracy_0k_50k_1 / total
    accuracy_0k_50k_2 = running_accuracy_0k_50k_2 / total

    accuracies = [accuracy_50k_0k, accuracy_40k_10k, accuracy_30k_20k, accuracy_20k_30k, accuracy_10k_40k,
                  accuracy_0k_50k_1, accuracy_0k_50k_2]

    print(accuracies)

    for i in range(len(model_list)):
        accuracies[i] = float('{:.3f}'.format(accuracies[i]))

    return matrices, accuracies


def write_matrices(matrix, accuracies, header):
    with open('confusion_matrices.csv', 'a') as file:
        write = writer(file)
        write.writerow([header])
        for i in range(len(matrix)):
            write.writerow([model_list[i]])
            write.writerow(['overall accuracy {}'.format(accuracies[i])])
            for j in range(len(classes)):
                write.writerow(matrix[i][j])
            write.writerow('')
        write.writerow('')
        file.close()


def write_accuracies(accuracies, noise_level):
    all_accuracies.append(noise_level)
    for i in range(len(accuracies)):
        all_accuracies.append('{}: {}'.format(model_list[i], accuracies[i]))
    all_accuracies.append('')


matrices_original, current_accuracies = evaluate(original_loader)
write_matrices(matrices_original, current_accuracies, 'no noise')
write_accuracies(current_accuracies, 'no noise')
print('no noise done')
all_accuracies.append('')


matrices_noise_1_1, current_accuracies = evaluate(noise_1_1_loader)
write_matrices(matrices_noise_1_1, current_accuracies, 'gaussian 1')
write_accuracies(current_accuracies, 'gaussian 1')
print('gaussian 1 done')

matrices_noise_1_2, current_accuracies = evaluate(noise_1_2_loader)
write_matrices(matrices_noise_1_2, current_accuracies, 'gaussian 2')
write_accuracies(current_accuracies, 'gaussian 2')
print('gaussian 2 done')

matrices_noise_1_3, current_accuracies = evaluate(noise_1_3_loader)
write_matrices(matrices_noise_1_3, current_accuracies, 'gaussian 3')
write_accuracies(current_accuracies, 'gaussian 3')
print('gaussian 3 done')

matrices_noise_1_4, current_accuracies = evaluate(noise_1_4_loader)
write_matrices(matrices_noise_1_4, current_accuracies, 'gaussian 4')
write_accuracies(current_accuracies, 'gaussian 4')
print('gaussian 4 done')

matrices_noise_1_5, current_accuracies = evaluate(noise_1_5_loader)
write_matrices(matrices_noise_1_5, current_accuracies, 'gaussian 5')
write_accuracies(current_accuracies, 'gaussian 5')
print('gaussian 5 done')

matrices_noise_1_6, current_accuracies = evaluate(noise_1_6_loader)
write_matrices(matrices_noise_1_6, current_accuracies, 'gaussian 6')
write_accuracies(current_accuracies, 'gaussian 6')
print('gaussian 6 done')
all_accuracies.append('')


matrices_noise_2_1, current_accuracies = evaluate(noise_2_1_loader)
write_matrices(matrices_noise_2_1, current_accuracies, 'speckle 1')
write_accuracies(current_accuracies, 'speckle 1')
print('speckle 1 done')

matrices_noise_2_2, current_accuracies = evaluate(noise_2_2_loader)
write_matrices(matrices_noise_2_2, current_accuracies, 'speckle 2')
write_accuracies(current_accuracies, 'speckle 2')
print('speckle 2 done')

matrices_noise_2_3, current_accuracies = evaluate(noise_2_3_loader)
write_matrices(matrices_noise_2_3, current_accuracies, 'speckle 3')
write_accuracies(current_accuracies, 'speckle 3')
print('speckle 3 done')

matrices_noise_2_4, current_accuracies = evaluate(noise_2_4_loader)
write_matrices(matrices_noise_2_4, current_accuracies, 'speckle 4')
write_accuracies(current_accuracies, 'speckle 4')
print('speckle 4 done')

matrices_noise_2_5, current_accuracies = evaluate(noise_2_5_loader)
write_matrices(matrices_noise_2_5, current_accuracies, 'speckle 5')
write_accuracies(current_accuracies, 'speckle 5')
print('speckle 5 done')

matrices_noise_2_6, current_accuracies = evaluate(noise_2_6_loader)
write_matrices(matrices_noise_2_6, current_accuracies, 'speckle 6')
write_accuracies(current_accuracies, 'speckle 6')
print('speckle 6 done')
all_accuracies.append('')


matrices_noise_3_1, current_accuracies = evaluate(noise_3_1_loader)
write_matrices(matrices_noise_3_1, current_accuracies, 'salt & pepper 1')
write_accuracies(current_accuracies, 'snp 1')
print('salt & pepper 1 done')

matrices_noise_3_2, current_accuracies = evaluate(noise_3_2_loader)
write_matrices(matrices_noise_3_2, current_accuracies, 'salt & pepper 2')
write_accuracies(current_accuracies, 'snp 2')
print('salt & pepper 2 done')

matrices_noise_3_3, current_accuracies = evaluate(noise_3_3_loader)
write_matrices(matrices_noise_3_3, current_accuracies, 'salt & pepper 3')
write_accuracies(current_accuracies, 'snp 3')
print('salt & pepper 3 done')

matrices_noise_3_4, current_accuracies = evaluate(noise_3_4_loader)
write_matrices(matrices_noise_3_4, current_accuracies, 'salt & pepper 4')
write_accuracies(current_accuracies, 'snp 4')
print('salt & pepper 4 done')

matrices_noise_3_5, current_accuracies = evaluate(noise_3_5_loader)
write_matrices(matrices_noise_3_5, current_accuracies, 'salt & pepper 5')
write_accuracies(current_accuracies, 'snp 5')
print('salt & pepper 5 done')

matrices_noise_3_6, current_accuracies = evaluate(noise_3_6_loader)
write_matrices(matrices_noise_3_6, current_accuracies, 'salt & pepper 6')
write_accuracies(current_accuracies, 'snp 6')
print('salt & pepper 6 done')


with open(r'overall_accuracies.txt', 'w') as file:
    for i in all_accuracies:
        file.write('{}\n'.format(i))
