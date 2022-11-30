from time import time

import numpy as np

from cuml.manifold import TSNE

import matplotlib.pyplot as plt

activation_data = np.load('data/activations_0.npy')
#activation_data =activation_data[:10000, :]

qvalues = np.load('data/qvalues_0.npy')
c = np.max(qvalues, axis=1)

print("Loaded")

tsne = TSNE(n_components=2, perplexity=50, n_neighbors=300, verbose=1, random_state=time())

print("Running tSNE")
fitted_data = tsne.fit_transform(activation_data)

print("tSNE done")

plt.scatter(fitted_data[:,0], fitted_data[:,1], s=4, c=c, cmap='plasma')

cbar = plt.colorbar()
cbar.set_label('Estimated Value')

plt.show()
