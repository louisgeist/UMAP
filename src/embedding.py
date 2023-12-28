import torch
from fuzzy_set import local_fuzzy_simplicial_set

import torch.nn as nn

def spectral_embedding(top_rep, embedding_dimension):
	"""
	Spectral embedding proposed in McInnes et al. (2020).

	Used for initialisation of the embedding optimization
	"""

	A = top_rep
	n = A.shape[0]
	D = torch.diag(A @ torch.ones(n))
	L = torch.sqrt(D) @ (D - A) @ torch.sqrt(D)

	eigenvalues , eigenvectors = torch.linalg.eig(L)
	eigenvalues, eigenvectors = torch.abs(eigenvalues), torch.abs(eigenvectors) # L is a real symetric matrix, then the eigenvalues are real

	Y = eigenvectors[torch.argsort(eigenvalues)[1:embedding_dimension+1]]

	return Y


X = torch.randn(10, 6)
A = local_fuzzy_simplicial_set(X,n_neighbors = 4)
B = A + A.transpose(0,1) - A * A.transpose(0,1)
spectral_embedding(B, embedding_dimension = 2)

def optimize_embedding(top_rep, Y, min_dist, n_epeochs):

	alpha = 1

	# fit phi from psi (defined by min_dist)

	def psi(x,y):
		norm2 = torch.sqrt(torch.sum((x-y)**2))

		if norm2 <= min_dist : 
			return 1
		else :
			return torch.exp(-(norm2 - min_dist))


	class phi_class(nn.Module):
		def __init__(self):
			super().__init__()
			self.a = nn.Parameter(torch.ones(1))
			self.b = nn.Parameter(torch.ones(1))

		def forward(self,x,y):
			return (1 + self.a * (torch.sum((x-y)**2))**self.a)**(-1)

	phi = phi_class()


	
	return(phi(torch.ones(2),2*torch.ones(2)))

print(optimize_embedding(B , None, 1, 1))



