# UMAP
## Introduction

Our report is available report [here](https://github.com).

![]

## Reproduce the results
All the python code available in the src folder.

The files dedicated to the reimplementation of UMAP are as follows:
- knn.py
- smooth_knn_dist.py
- fuzzy_set.py
- embedding.py
- umap.py

The *umap.py* file is the main one; you only have to import the function named *umap* from that file in order to use our reimplementation on your data.

The *train_.....py* (*mnist*, *interlaced_rings*) files apply the *umap* function to the specified data and plots the result in a 2D space.

The *evaluation.py* file proposes evualation of the model based on the [AMI score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html). This score is based on a KMeans in order to assess the clustering ability of the model. We thus display also the compressed data next to the KMeans clusters. The *visualisation.py* file displays all the reduced data that have been saved in the folder data_reduced/MNIST by a run of *train_mnist.py*.