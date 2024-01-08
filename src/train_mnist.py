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

for i,x in enumerate(train_dataset) :
	X.append(x[0])
	labels.append(float(x[1]/10) )

	if i > 1000:
		break

X = torch.stack(tuple(X))


print("Begin")
Y, Y_initial = UMAP(X, 100, 2, 0.01, 400)


plt.figure()
plt.scatter(Y[:,0],Y[:,1], c = labels)
plt.title("Reduction from 3D to 2D")
plt.show()

