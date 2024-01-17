# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:44:50 2024

@author: maeld
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from umap import UMAP

def torus(N,r) :
  """
  simulation of a noisy doble torus
  """

  simu_x = []
  simu_y = []
  simu_z = []

  for i in range(N):
    U_x = 0
    U_y = 0
    U_z = r+1
    while (U_x*(U_x-1)**2*(U_x-2) + (U_y/5)**2)**2 + (U_z/15)**2 > r:
      U_x = 4*np.random.rand()-2
      U_y = 8*np.random.rand()-4
      U_z = 4*np.random.rand()-2
    simu_x.append(U_x + np.random.normal(0,0.1))
    simu_y.append(U_y + np.random.normal(0,0.1))
    simu_z.append(U_z + np.random.normal(0,0.1))

  data = torch.zeros(N,3)
  data[:,0] =  torch.tensor(simu_x)
  data[:,1] =  torch.tensor(simu_y)
  data[:,2] =  torch.tensor(simu_z)

  return data


r = 0.001
N = 2000
data = torus(N,r)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = data[:, 0].numpy()
y = data[:, 1].numpy()
z = data[:, 2].numpy()

ax.scatter(x, y, z, marker='o', s=10)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

plt.figure()
plt.scatter(x,y, s=10)
plt.show()


n_epochs = 400
n_neighbors = 100
min_dist = 0.05
Y , Y_initial = UMAP(data, n_neighbors = n_neighbors, embedding_dimension = 2, min_dist = min_dist, n_epochs = n_epochs)


plt.figure()
plt.scatter(Y[:,0],Y[:,1], s = 10)
plt.title("Reduction from 3D to 2D")
plt.show()
