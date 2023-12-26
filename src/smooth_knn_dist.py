
import torch
import math

def knn(X, k):
    """
    X is a tensor of size (n,d)
    k is the number of nearest neighbors we are looking for

    X

    """
    n = X.shape[0]

    dots = X @ X.T
    norms = (X**2).sum(axis=1)

    dist = torch.sqrt(norms.view(n, 1) + norms.view(1, n) - 2 * dots)

    knn_dists, knn = dist.topk(dim=1, largest=False, k=k)

    return knn, knn_dists

    # |x-y|^2 = |x|^2 - 2<x,y> + |y|^2

    # return knn, knn_dists


X = torch.randn(4, 2)
print(X)
print(knn(X, 2))


def smooth_knn_dist(knn,knn_dist,err):
    
    """
    err is the stopping error
    
    knn and knn_dist are tensors of size (n,k) with k the number of neighbors


    """

    k = knn_dist.shape[1]
    n = knn_dist.shape[0]
    
    sigma_tens = torch.zeros(n)
    
    for i in range(n):
        dist_xi = knn_dist[[i]]
        p = dist_xi[0][1].item()
        d1 = dist_xi[0][2].item()
        b = (d1-p)/100
        a = (d1-p)/math.log(n/(math.log(n)/math.log(2)))
        sigma = (a+b)/2
        mem = b
        while abs(mem-sigma) > err:
            sum_exp_sigma = torch.exp(-((dist_xi-p)/sigma)).sum()
            if sum_exp_sigma<math.log(k)/math.log(2):
                a = sigma
            else:
                b = sigma
            mem = sigma
            sigma = (a+b)/2
        sigma_tens[i] = sigma
    return sigma_tens

X = torch.randn(30,10)
res = smooth_knn_dist(knn(X, 6)[0], knn(X,6)[1],10**(-5)) 
print(res)