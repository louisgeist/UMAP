# -*- coding: utf-8 -*-
"""

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from umap import UMAP

def torus(N,r,R) :
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

     while (R-np.sqrt(U_x**2+U_y**2))**2 + U_z**2 > r**2:
       U_x = 8*np.random.rand()-4
       U_y = 8*np.random.rand()-4
       U_z = 2*np.random.rand()-1
     simu_x.append(U_x + np.random.normal(0,0.5))
     simu_y.append(U_y + np.random.normal(0,0.5))
     simu_z.append(U_z + np.random.normal(0,0.5))

  data = torch.zeros(N,3)
  data[:,0] =  torch.tensor(simu_x)
  data[:,1] =  torch.tensor(simu_y)
  data[:,2] =  torch.tensor(simu_z)

  return data


R = 3
r = 1
  
N = 1000
data = torus(N,r,R)
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


#UMAP
n_epochs = 400
n_neighbors = 100
min_dist = 0.05
Y , Y_initial = UMAP(data, n_neighbors = n_neighbors, embedding_dimension = 2, 
   min_dist = min_dist, n_epochs = n_epochs)


plt.figure()
plt.scatter(Y[:,0],Y[:,1], s = 1)
plt.title("Reduction from 3D to 2D")
plt.show()
