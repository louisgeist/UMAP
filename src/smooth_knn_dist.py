
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



def smooth_knn_dist(knn,knn_dist,err):
    
    """
    err is the stopping error
    
    knn and knn_dist are tensors of size (n,k) with k the number of neighbors


    """

    k = knn_dist.shape[1]
    n = knn_dist.shape[0]
    
    sigma_tens = torch.zeros(n)
    
    for i in range(n):
        dist_xi = knn_dist[[i]][0][1:]
        p = dist_xi[0].item()
        d_min = dist_xi[1].item()
        d_max = dist_xi[k-2].item()
        
        if k>8:
            b = d_max-p
        else: #si k<8 on est pas assuré d'avoir une quantité positive avec d1-p
            b = 10*(d_max-p) #en multipliant par 100 on est de nouveau sur d'avoir qqchose de >0
            
        a = (d_min-p)/math.log(k/(math.log(k)/math.log(2)))
        
        #vérif conditions initiales --> à enlever
        test_a = torch.exp(-((dist_xi-p)/a)).sum() -math.log(k)/math.log(2)
        test_b = torch.exp(-((dist_xi-p)/b)).sum() -math.log(k)/math.log(2)
        if test_a>0 or test_b<0:
            print('WARNING')
            
        sigma = (a+b)/2
        mem = b
        while abs(mem-sigma) > err:
            test = torch.exp(-((dist_xi-p)/sigma)).sum() - math.log(k)/math.log(2)
            if test<0:
                a = sigma
            else:
                b = sigma
            mem = sigma
            sigma = (a+b)/2
        sigma_tens[i] = sigma
    return sigma_tens

X = torch.randn(30,8)
res = smooth_knn_dist(knn(X, 5)[0], knn(X,5)[1],10**(-5)) 