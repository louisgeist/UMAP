import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from umap import UMAP


batch_size = 32
n_epoch = 400
n_neighbors = 100
embedding_dimension = 2
min_dist = 0.05

n_points = 1000

trans2D = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)),
        ])

flatten = transforms.Lambda(lambda x: x.view(-1))

train_dataset = datasets.MNIST(root='../data', train = True, transform=trans2D, download=True)
train_dataset.transform = transforms.Compose([trans2D, flatten])


X = []
labels = []


for i,x in enumerate(train_dataset) :
	X.append(x[0])
	labels.append(x[1])

	if i > n_points :
		break

X = torch.stack(tuple(X))

np.save(f"../data_reduced/MNIST/{n_points}pts_MNISTraw.npy",X)
np.save(f"../data_reduced/MNIST/{n_points}pts_MNISTraw_labels.npy",labels)

print("Begin")
Y, Y_initial = UMAP(X, n_neighbors, embedding_dimension, min_dist, n_epoch)

np.save(f"../data_reduced/MNIST/{n_points}pts_{n_epoch}e_{n_neighbors}knn_{min_dist}min.npy",Y)
np.save(f"../data_reduced/MNIST/{n_points}pts_{n_epoch}e_{n_neighbors}knn_{min_dist}min_labels.npy",labels)

fig, ax = plt.subplots()

scatter = ax.scatter(Y[:,0],Y[:,1], c = labels, cmap = "Set3", label = "Data Points")
legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.title(f"{n_points} samples of MNIST (28x28) represented in 2D space")


plt.axis('off')
plt.show()

