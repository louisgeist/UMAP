# UMAP
## Introduction

The goal of the project was to understand the [paper introducing UMAP](https://arxiv.org/abs/1802.03426) written by McInnes *et al.* and to implement the algorithm from scratch. Our [report](https://github.com/louisgeist/UMAP/blob/main/report.pdf) synthesizes the theoretical foundations of the UMAP algorithm and its practical implementation. It is accessible for a Master's student in applied mathematics. This GitHub repository contains our reimplementation of the algorithm. A section below explains how to reproduce our results.

![Figure](https://github.com/louisgeist/UMAP/blob/main/figure/1e%5E4samples_400e_0.05mindist.png)
The figure above displays the results of the UMAP algorithm applied to 10,000 samples from the MNIST dataset, with the following parameters: 100 nearest neighbors, 0.05 minimal distance, and 400 training epochs.

## Reproduce the results
All the python code is available in the src folder.

The files dedicated to the reimplementation of UMAP are as follows:
- knn.py
- smooth_knn_dist.py
- fuzzy_set.py
- embedding.py
- umap.py

The *umap.py* file is the main one; you only have to import the function named *umap* from that file in order to use our reimplementation on your data.

The *train_.....py* (*mnist*, *interlaced_rings*) files apply the *umap* function to the specified data and plots the result in a 2D space.

The *evaluation.py* file proposes evualation of the model based on the [AMI score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html). This score is based on a KMeans in order to assess the clustering ability of the model. We thus display also the compressed data next to the KMeans clusters. The *visualisation.py* file displays all the reduced data that have been saved in the folder data_reduced/MNIST by a run of *train_mnist.py*.
