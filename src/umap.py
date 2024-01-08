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


