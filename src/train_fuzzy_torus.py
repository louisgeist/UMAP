# -*- coding: utf-8 -*-
"""

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from umap import UMAP

def torus(N,r,a,R) :
  """
  simulation of a noisy doble torus
  """

  
  simu_x = []
  simu_y = []
  simu_z = []
  for i in range(N):
     x = 0
     y = 0
     z = r+1
    #to ensure differentiability, we use this large equation
     while -a**2 + ((-r**2 + R**2)**2 - 2*(r**2 + R**2)*((-r - R + x)**2 + y**2) + 2*(-r**2 + R**2)*z**2 + ((-r - R + x)**2 + y**2 + z**2)**2)*((-r**2 + R**2)**2 - 2*(r**2 + R**2)*((r + R + x)**2 + y**2) + 2*(-r**2 + R**2)*z**2 + ((r + R + x)**2 + y**2 + z**2)**2)>0.01:
        x = 8*np.random.rand()-4
        y = 5*np.random.rand()-2.5
        z = 2*np.random.rand()-1
     simu_x.append(x)
     simu_y.append(y)
     simu_z.append(z)


  data = torch.zeros(N,3)
  data[:,0] =  torch.tensor(simu_x)
  data[:,1] =  torch.tensor(simu_y)
  data[:,2] =  torch.tensor(simu_z)

  return data


R = 1.5
r = 0.3
a = np.sqrt(2)
  
N = 10**2
data = torus(N,r,a,R)
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
N = 10**4
data = torus(N,r,a,R)
n_epochs = 400
n_neighbors = 100
min_dist = 0.05
Y , Y_initial = UMAP(data, n_neighbors = n_neighbors, embedding_dimension = 2, min_dist = min_dist, n_epochs = n_epochs)


plt.figure()
plt.scatter(Y[:,0],Y[:,1], s = 10)
plt.title("Reduction from 3D to 2D")
plt.show()
