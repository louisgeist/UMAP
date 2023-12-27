import torch


def knn(X, k):
    """
    X is a tensor of size (n,d)
    k is the number of nearest neighbors we are looking for

    X

    """
    n = X.shape[0]

    dots = X @ X.T
    norms = (X**2).sum(axis=1)

    dist = torch.sqrt((norms.view(n, 1) + norms.view(1, n) - 2 * dots).clamp_min(0) )
    print("dist dans knn", dist)

    knn_dists, knn = dist.topk(dim=1, largest=False, k=k)

    return knn, knn_dists

    # |x-y|^2 = |x|^2 - 2<x,y> + |y|^2

    # return knn, knn_dists


# X = torch.randn(10, 6)
# print(X)
# print(knn(X, 4))
