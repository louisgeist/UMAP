import torch

from umap import UMAP

def interlaced_rings(n_samples = 100):
	"""
	Returns an artificial dataset composed of 2 rings that are interlaced
	"""

	data = torch.zeros(2*n_samples,3)

	t = torch.linspace(0, 1, n_samples)

	x = torch.cos(2*torch.pi*t).view(-1, 1)
	y = torch.sin(2*torch.pi*t).view(-1, 1)

	data[:n_samples, :2] = torch.cat([x, y], dim=1)

	x = 3/2 + x
	data[n_samples:, 1:] = torch.cat([x, y], dim=1)

	return data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
n_samples = 100
colors = ['blue'] * n_samples + ['red'] * n_samples

X = interlaced_rings(n_samples)

# dataset displayed in 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = X[:, 0].numpy()
y = X[:, 1].numpy()
z = X[:, 2].numpy()

ax.scatter(x, y, z, c=colors, marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

n_epochs = 400
n_neighbors = 100
min_dist = 0.05
Y , Y_initial = UMAP(X, n_neighbors = n_neighbors, embedding_dimension = 2, min_dist = min_dist, n_epochs = n_epochs)


plt.figure()
plt.scatter(Y[:,0],Y[:,1], c = colors)
plt.title("Reduction from 3D to 2D")
plt.show()

plt.figure()
plt.scatter(Y_initial[:,0],Y_initial[:,1], c = colors)
plt.title("Spectral embedding - initilialization for Y")
plt.show()
