import os
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

directory = '../data_reduced/MNIST'

def clustering_evaluation(Y,labels, plot = True):

	clusterer = KMeans(10, n_init = 'auto')
	clusterer.fit(Y)
	pedicted_labels = clusterer.predict(Y)

	print("Adjusted random score :", adjusted_rand_score(labels, pedicted_labels))
	# Plot
	if plot :
		fig, ax = plt.subplots(1, 2, figsize =(10, 4))
		scatter_true = ax[0].scatter(Y[:,0],Y[:,1], c = labels, cmap = "Set3", label = "Data Points", s=15)
		legend_true = ax[0].legend(*scatter_true.legend_elements(), loc="lower left", title="Classes")
		ax[0].add_artist(legend_true)
		ax[0].axis('off')

		scatter_predicted = ax[1].scatter(Y[:,0],Y[:,1], c = pedicted_labels, cmap = "Set1", label = "Data Points", s=15)
		ax[1].axis('off')

		plt.tight_layout()
		plt.show()


#### ----- visualisation loop ----- ####
index = 1
for file_name in os.listdir(directory):
	plot = True

	if file_name == '.DS_Store': continue
	if "labels" in file_name : continue

	if "raw" in file_name:
		plot = False
	
	print(f"Information about the set n°{index} : ",file_name)

	complete_path = "{}/{}".format(directory, file_name)

	Y = np.load(complete_path)
	labels = np.load(complete_path[:-4]+"_labels"+complete_path[-4:])

	clustering_evaluation(Y, labels, plot)
	index += 1

	print("")
	