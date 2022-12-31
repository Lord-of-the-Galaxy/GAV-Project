import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
from control_buttons import add_buttons as add_control_buttons
from control_buttons import GAME1, GAME2, GAME1_SHORT, GAME2_SHORT, INDEX
from time import time
import numpy as np
from cuml import TSNE
from turtle import color
import numpy as np
from matplotlib.widgets import Button

from samdp import samdp_search

class VIS_TOOL:

    def __init__(self):


        # 1. Constants
        self.perplexity = 60

        # 2. Plots
        self.fig = plt.figure('tSNE')
        
        

        # 2.1 t-SNE
        self.ax_tsne = plt.subplot2grid((30,40),(0,0), rowspan=30, colspan=30)
        self.ax_screen = plt.subplot2grid((30,40),(15,33), rowspan=15, colspan=7)
        self.tsne_scat = self.ax_tsne.scatter([0],[0])
        self.cbar = self.fig.colorbar(self.tsne_scat)
        self.samdp_lines = self.ax_tsne.plot()
        self.update_tSNE(GAME1)


        '''
        # 2.3 gradient image (saliency map)
        self.ax_state = plt.subplot2grid((3,5),(2,4), rowspan=1, colspan=1)

        self.stateplot = self.ax_state.imshow(self.states[self.ind], interpolation='none', cmap='gray',picker=5)
        self.ax_state.set_xticklabels([])
        self.ax_state.set_yticklabels([])
        '''

        # 3. Control buttons
        add_control_buttons(self)
        self.fig.canvas.mpl_connect('pick_event', self.on_scatter_pick)

   

    def update_tSNE(self,game):
        activation_data = np.load(f'data/{game}/activations_{INDEX}.npy')
        qvalues = np.load(f'data/{game}/qvalues_{INDEX}.npy')
        self.values = np.max(qvalues, axis=1)
        self.rewards = np.load(f'data/{game}/rewards_{INDEX}.npy')
        self.screens = np.load(f'data/{game}/images_{INDEX}.npy')
        print("Loaded")


        self.pnt_size = 5
        self.ind      = 0
        self.prev_ind = 0

        tsne = TSNE(n_components=2, perplexity=self.perplexity, n_neighbors=100+self.perplexity*3, verbose=1, random_state=time(),method = 'fft')
        print("Running tSNE")
        self.data_t = tsne.fit_transform(activation_data)
        print("tSNE done")

        self.num_points = self.data_t.shape[0]

        #self.tsne_scat.remove()
        self.ax_tsne.cla()
        self.tsne_scat = self.ax_tsne.scatter(self.data_t[:,0],
                                     self.data_t[:,1],
                                     s= np.ones(self.num_points)*self.pnt_size, c = self.values, cmap='gist_rainbow',picker=5)

        self.ax_tsne.set_xticklabels([])
        self.ax_tsne.set_yticklabels([])

        #colorbar
        self.cbar.remove()
        cb_axes = plt.axes([0.04,0.11,0.01,0.78])
        self.cbar = self.fig.colorbar(self.tsne_scat, cax=cb_axes)
        self.cbar.set_label("Estimated value")

        self.screenplot = self.ax_screen.imshow(self.screens[self.ind], interpolation='none')

        self.ax_screen.set_xticklabels([])
        self.ax_screen.set_yticklabels([])
        #self.ax_tsne.autoscale(False)
        self.fig.canvas.draw()
        
        self.prev_color = self.tsne_scat.get_facecolors()[self.prev_ind]

    def on_scatter_pick(self,event):
        self.ind = event.ind[0]
        self.update_plot()
        self.prev_ind = self.ind

    def update_plot(self):
        self.screenplot.set_array(self.screens[self.ind])
        #self.stateplot.set_array(self.states[self.ind])
        sizes = self.tsne_scat.get_sizes()
        sizes[self.ind] = 100
        sizes[self.prev_ind] = self.pnt_size
        self.tsne_scat.set_sizes(sizes)
        
        colors = self.tsne_scat.get_facecolors()
        colors[self.prev_ind] = self.prev_color
        self.prev_color = colors[self.ind]
        colors[self.ind] = np.array([0, 0, 0, 1])
        self.tsne_scat.set_facecolors(colors)
        
        self.tsne_scat.set_color(colors)
        
        self.fig.canvas.draw()
        #print 'chosen point: %d' % self.ind

    def samdp(self):
        
        nc, coords, v_samdp, labels, P, vmse, inertia, entropy = samdp_search(self.data_t, self.values, self.rewards, 16, 30, 2, 5)
        
        self.ax_tsne.cla()
        
        self.tsne_scat = self.ax_tsne.scatter(self.data_t[:,0],
                                     self.data_t[:,1],
                                     s= np.ones(self.num_points)*self.pnt_size, c = self.values, cmap='gist_rainbow',picker=5)

        self.ax_tsne.set_xticklabels([])
        self.ax_tsne.set_yticklabels([])
        
        for i in range(nc):
            for j in range(nc):
                if P[i, j] > 0.05:
                    self.ax_tsne.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], c='black', linewidth=P[i, j]*5, alpha=0.5)

        self.ax_tsne.scatter(coords[:, 0], coords[:, 1], s=100, c='black', marker='D')
        

        #self.ax_tsne.autoscale(False)
        self.fig.canvas.draw()


    def clear_samdp(self):
        
        self.ax_tsne.cla()
        self.tsne_scat = self.ax_tsne.scatter(self.data_t[:,0],
                                     self.data_t[:,1],
                                     s= np.ones(self.num_points)*self.pnt_size, c = self.values, cmap='gist_rainbow',picker=5)

        self.ax_tsne.set_xticklabels([])
        self.ax_tsne.set_yticklabels([])

        #self.ax_tsne.autoscale(False)
        self.fig.canvas.draw()


    def show(self):
        plt.show(block=True)



obj = VIS_TOOL()
obj.show()




