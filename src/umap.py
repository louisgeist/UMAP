import torch

from fuzzy_set import local_fuzzy_simplicial_set
from embedding import spectral_embedding
from embedding import optimize_embedding

import time

def UMAP(X, n_neighbors, embedding_dimension, min_dist, n_epochs):
	t1 = time.time()

	A = local_fuzzy_simplicial_set(X,n_neighbors)
	t2 = time.time()


	B = A + A.transpose(0,1) - A * A.transpose(0,1)

	#Y = torch.randn(B.shape[0], embedding_dimension) # arbitrary initialization
	Y = spectral_embedding(B, embedding_dimension)
	Y_initial = Y.detach().numpy().copy() 
	t3 = time.time()

	Y = optimize_embedding(B, Y, min_dist, n_epochs)
	t4 = time.time()

	print("Time to compute the local fuzzy sets : ", t2-t1)
	print("Time for computing spectral embedding : ", t3-t2)
	print("Time for embedding optimization : ", t4-t3)


	return Y.detach().numpy() , Y_initial


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

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = X[:, 0].numpy()
# y = X[:, 1].numpy()
# z = X[:, 2].numpy()

# ax.scatter(x, y, z, c=colors, marker='o')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()

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


