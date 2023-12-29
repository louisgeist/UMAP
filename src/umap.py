import torch

from fuzzy_set import local_fuzzy_simplicial_set
from embedding import spectral_embedding
from embedding import optimize_embedding

def UMAP(X, n_neighbors, embedding_dimension, min_dist, n_epochs):

	A = local_fuzzy_simplicial_set(X,n_neighbors)

	B = A + A.transpose(0,1) - A * A.transpose(0,1)

	#Y = torch.randn(B.shape[0], embedding_dimension) # arbitrary initialization
	Y = spectral_embedding(B, embedding_dimension)
	Y = optimize_embedding(B, Y, min_dist, n_epochs)

	return Y.detach().numpy()


def interlaced_ring(n_samples = 100):
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

X = interlaced_ring(n_samples)
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


Y = UMAP(X, n_neighbors = 100, embedding_dimension = 2, min_dist = 0.01, n_epochs = 20)

plt.figure()
plt.scatter(Y[:,0],Y[:,1], c = colors)
plt.show()


