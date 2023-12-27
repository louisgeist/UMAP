
import torch
import math

from knn import knn


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
        rho = dist_xi[0].item()
        d_min = dist_xi[1].item()
        d_max = dist_xi[k-2].item()
        
        if k>8:
            b = d_max-rho
        else: #si k<8 on est pas assuré d'avoir une quantité positive avec d1-p
            b = 10*(d_max-rho) #en multipliant par 10 on est de nouveau sur d'avoir qqchose de >0
            
        a = (d_min-rho)/math.log((k-1)/((math.log(k)/math.log(2))-1))
        
        #vérif conditions initiales --> à enlever
        test_a = torch.exp(-((dist_xi-rho)/a)).sum() -math.log(k)/math.log(2)
        test_b = torch.exp(-((dist_xi-rho)/b)).sum() -math.log(k)/math.log(2)
        if test_a>0 or test_b<0:
            print('WARNING')
            print('a',test_a)
            print('b',test_b)
            
        sigma = (a+b)/2
        mem = b
        while abs(mem-sigma) > err:
            test = torch.exp(-((dist_xi-rho)/sigma)).sum() - math.log(k)/math.log(2)
            if test<0:
                a = sigma
            else:
                b = sigma
            mem = sigma
            sigma = (a+b)/2
        sigma_tens[i] = sigma
    return sigma_tens

# X = torch.randn(30,8)
# res = smooth_knn_dist(knn(X, 5)[0], knn(X,5)[1],10**(-5)) 
# print(res)
