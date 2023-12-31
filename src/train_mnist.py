import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib

from umap import UMAP

batch_size = 20

trans2D = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)),
        ])

flatten = transforms.Lambda(lambda x: x.view(-1))

train_dataset = datasets.MNIST(root='../data', train = True, transform=trans2D, download=True)
train_dataset.transform = transforms.Compose([trans2D, flatten])
# dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


X = []
labels = []

n_points = 1000

for i,x in enumerate(train_dataset) :
	X.append(x[0])
	labels.append(x[1])

	if i > n_points :
		break

X = torch.stack(tuple(X))


print("Begin")
Y, Y_initial = UMAP(X, 60, 2, 0.05, 400)

fig, ax = plt.subplots()

scatter = ax.scatter(Y[:,0],Y[:,1], c = labels, cmap = "Set3", label = "Data Points")
legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.title(f"{n_points} samples of MNIST (28x28) represented in 2D space")


plt.axis('off')
plt.show()

