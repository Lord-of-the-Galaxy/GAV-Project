from samdp import samdp_search
from matplotlib.widgets import Button
from turtle import color
from cuml import TSNE
from time import time
from control_buttons import GAME1, SEED1
from control_buttons import add_buttons as add_control_buttons
from matplotlib.widgets import Button, Slider, CheckButtons
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class VIS_TOOL:

    def __init__(self):

        # 1 Constants
        self.perplexity = 60

        # 2 Plots
        self.fig = plt.figure('tSNE + SAMDP')

        # 2.1 Event Handlers
        self.fig.canvas.mpl_connect('draw_event', self.on_draw)
        self.fig.canvas.mpl_connect('pick_event', self.on_scatter_pick)

        # 2.2 t-SNE
        self.ax_tsne = plt.subplot2grid(
            (30, 40), (0, 0), rowspan=30, colspan=30)

        self.tsne_scat = self.ax_tsne.scatter(
            [0], [0])  # just for the colorbar
        self.cbar = self.fig.colorbar(self.tsne_scat)

        # 2.3 State image and pointer
        self.idx = 0
        self.ax_image = plt.subplot2grid(
            (30, 40), (15, 33), rowspan=15, colspan=7)
        self.image = self.ax_image.imshow([[[0, 0, 0]]], interpolation='none')
        self.ax_image.set_xticklabels([])
        self.ax_image.set_yticklabels([])

        self.point, = self.ax_tsne.plot([0], [0], animated=True, linestyle="",
                                        marker="o", markersize=10, markerfacecolor="r", markeredgecolor="k")

        # 2.4 SAMDP
        self.samdp_done = False
        self.nc = 0
        self.coords = np.array([[]])
        self.P = np.array([[]])

        # 3 Actually compute and draw the tSNE
        self.update_tSNE(GAME1, SEED1)

        # 4 Control buttons
        add_control_buttons(self)

    def on_draw(self, event):
        # grab tSNE plot on every draw
        self.tsne_plot = self.fig.canvas.copy_from_bbox(self.ax_tsne.bbox)
        self.draw_animated()

    def draw_animated(self):
        self.ax_tsne.draw_artist(self.point)
        self.ax_image.draw_artist(self.image)

        if self.samdp_done:
            # draw in the SAMDP transitions
            for i in range(self.nc):
                for j in range(self.nc):
                    if self.P[i, j] > 0.02:
                        a, = self.ax_tsne.plot([self.coords[i, 0], self.coords[j, 0]],
                                               [self.coords[i, 1], self.coords[j, 1]], c='black', linewidth=self.P[i, j]*5, alpha=0.4, animated=True)
                        self.ax_tsne.draw_artist(a)

            # draw in SAMDP cluster locations
            a, = self.ax_tsne.plot(
                self.coords[:, 0], self.coords[:, 1], animated=True, linestyle="", markersize=12, markerfacecolor='black', markeredgecolor='black', marker='D')
            self.ax_tsne.draw_artist(a)

    def update_tSNE(self, game, seed):
        print("Loading...")
        activation_data = np.load(f'data/{game}/activations_{seed}.npy')
        qvalues = np.load(f'data/{game}/qvalues_{seed}.npy')
        self.values = np.max(qvalues, axis=1)
        self.rewards = np.load(f'data/{game}/rewards_{seed}.npy')
        self.images = np.load(f'data/{game}/images_{seed}.npy')
        print("Loaded")

        self.idx = 0
        self.samdp_done = False

        tsne = TSNE(n_components=2, perplexity=self.perplexity, n_neighbors=100 +
                    self.perplexity*3, verbose=1, random_state=time(), method='fft')
        print("Running tSNE...")
        self.data_t = tsne.fit_transform(activation_data)
        print("tSNE done")

        self.num_points = self.data_t.shape[0]

        # t-SNE plot
        self.ax_tsne.cla()
        self.tsne_scat = self.ax_tsne.scatter(self.data_t[:, 0], self.data_t[:, 1],
                                              s=5, c=self.values, cmap='gist_rainbow', picker=5)

        self.ax_tsne.set_xticklabels([])
        self.ax_tsne.set_yticklabels([])

        # colorbar
        self.cbar.remove()
        cb_axes = plt.axes([0.04, 0.11, 0.01, 0.78])
        self.cbar = self.fig.colorbar(self.tsne_scat, cax=cb_axes)
        self.cbar.set_label("Estimated value")

        # draw, and then save the tSNE plot to avoid redrawing when not updating the tSNE
        self.fig.canvas.draw()

        # self.tsne_plot = self.fig.canvas.copy_from_bbox(self.ax_tsne.bbox)

        # draw the currently selected point & update the image
        self.update_plot()

        # self.ax_tsne.autoscale(False)

    def on_scatter_pick(self, event):
        # print("Scatter pick")
        self.idx = event.ind[0]
        self.update_plot()

    def update_plot(self):
        # print("Update called")
        # update state image
        self.image.set_array(self.images[self.idx])

        # Update point
        self.point.set_data([self.data_t[self.idx, 0]],
                            [self.data_t[self.idx, 1]])
        self.point.set_markerfacecolor(
            self.tsne_scat.get_facecolors()[self.idx])

        # self.ax_tsne.cla()

        # restore the tSNE plot
        self.fig.canvas.restore_region(self.tsne_plot)

        # draw animated components
        self.draw_animated()

        # blit in the axes
        self.fig.canvas.blit(self.ax_tsne.bbox)
        self.fig.canvas.blit(self.ax_image.bbox)

    def samdp(self):

        if self.samdp_done:
            return

        self.nc, self.coords, v_samdp, labels, self.P, vmse, inertia, entropy = samdp_search(
            self.data_t, self.values, self.rewards, 12, 30, 2, 6)

        self.samdp_done = True

        # self.ax_tsne.cla()

        # self.tsne_scat = self.ax_tsne.scatter(self.data_t[:, 0],
        #                                       self.data_t[:, 1],
        #                                       s=np.ones(self.num_points)*self.pnt_size, c=self.values, cmap='gist_rainbow', picker=5)

        # self.ax_tsne.set_xticklabels([])
        # self.ax_tsne.set_yticklabels([])

        # restore the tSNE plot
        self.fig.canvas.restore_region(self.tsne_plot)

        # draw animated components
        self.draw_animated()

        # blit in the axes
        self.fig.canvas.blit(self.ax_tsne.bbox)
        self.fig.canvas.blit(self.ax_image.bbox)

    def clear_samdp(self):

        # self.ax_tsne.cla()
        # self.tsne_scat = self.ax_tsne.scatter(self.data_t[:, 0],
        #                                       self.data_t[:, 1],
        #                                       s=np.ones(self.num_points)*self.pnt_size, c=self.values, cmap='gist_rainbow', picker=5)

        # self.ax_tsne.set_xticklabels([])
        # self.ax_tsne.set_yticklabels([])

        self.samdp_done = False
        self.nc = 0
        self.coords = np.array([[]])
        self.P = np.array([[]])

        # restore the tSNE plot
        self.fig.canvas.restore_region(self.tsne_plot)

        # draw animated components
        self.draw_animated()

        # blit in the axes
        self.fig.canvas.blit(self.ax_tsne.bbox)
        self.fig.canvas.blit(self.ax_image.bbox)

    def show(self):
        plt.show(block=True)


obj = VIS_TOOL()
obj.show()
