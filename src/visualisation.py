import os
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

directory = '../data_reduced/MNIST'

plot_index = 1
for file_name in os.listdir(directory):
	if file_name == '.DS_Store': continue
	if "labels" in file_name : continue

	print(plot_index)
	print(f"Information about the plot n°{plot_index} : ",file_name)

	complete_path = "{}/{}".format(directory, file_name)

	Y = np.load(complete_path)
	labels = np.load(complete_path[:-4]+"_labels"+complete_path[-4:])

	fig, ax = plt.subplots()

	scatter = ax.scatter(Y[:,0],Y[:,1], c = labels, cmap = "Set3", label = "Data Points", s=15)
	legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
	ax.add_artist(legend1)


	plt.axis('off')
	plt.show()

	plot_index += 1