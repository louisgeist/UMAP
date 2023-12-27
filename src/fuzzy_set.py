import torch
import networkx as nx

from knn import knn
from smooth_knn_dist import smooth_knn_dist


def local_fuzzy_simplicial_set(X, n_neighbors):
	"""
	returns a matrix A of shape (n,n),

	where each line A[i,:] is the weights corresponding to the fs-set associated 
	to the knn of the point X indexed by i

	"""

	knn_ind, knn_dists = knn(X,n_neighbors)
	
	rho = knn_dists[:,1]
	sigma = smooth_knn_dist(knn_ind, knn_dists, err = 1e-6)

	n = X.shape[0]

	A = torch.zeros((n,n))

	for i,x in enumerate(X):

		weights = torch.zeros(n)
		weights[knn_ind[i]] = torch.exp(-(knn_dists[i]-rho[i]).clamp_min(0)/sigma[i])
		A[i,:] = weights

		A[i,i] = 0

	return A


# X = torch.randn(10, 6)
# res = local_fuzzy_simplicial_set(X,6)
# print(res)
