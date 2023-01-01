import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib import path
from matplotlib.patches import Polygon
from cuml.manifold import TSNE
from time import time
import pickle

GAME1 = 'MsPacman'
GAME2 = 'Breakout'
GAME3 = 'Seaquest'
GAME4 = 'SpaceInvaders'
GAME1_SHORT = 'Pacman'
GAME2_SHORT = 'Breakout'
GAME3_SHORT = 'Seaquest'
GAME4_SHORT = 'Space Inv'

SEED1 = 1672518751
SEED2 = 1672518751
SEED3 = 1672518751
SEED4 = 1672518751


def add_buttons(self):

    #############################
    # 1. play 1 b/w step
    #############################
    def BW(event):
        self.idx = (self.idx - 1) % self.num_points
        self.update_plot()

    self.ax_bw = plt.axes([0.75, 0.80, 0.09, 0.02])
    self.b_bw = Button(self.ax_bw, 'B/W')
    self.b_bw.on_clicked(BW)

    #############################
    # 2. play 1 f/w step
    #############################
    def FW(event):
        self.idx = (self.idx + 1) % self.num_points
        self.update_plot()

    self.ax_fw = plt.axes([0.85, 0.80, 0.09, 0.02])
    self.b_fw = Button(self.ax_fw, 'F/W')
    self.b_fw.on_clicked(FW)

    #############################
    # 1. play 10 b/w step
    #############################
    def BWT(event):
        self.idx = (self.idx - 10) % self.num_points
        self.update_plot()

    self.ax_bwt = plt.axes([0.75, 0.83, 0.09, 0.02])
    self.b_bwt = Button(self.ax_bwt, 'B/W X10')
    self.b_bwt.on_clicked(BWT)

    #############################
    # 2. play 10 f/w step
    #############################
    def FWT(event):
        self.idx = (self.idx + 10) % self.num_points
        self.update_plot()

    self.ax_fwt = plt.axes([0.85, 0.83, 0.09, 0.02])
    self.b_fwt = Button(self.ax_fwt, 'F/W X10')
    self.b_fwt.on_clicked(FWT)

    #############################
    # 3. Run tSNE on Game 1 again
    #############################

    def g1_repeat(event):
        self.update_tSNE(GAME1, SEED1)

    self.ax_g1 = plt.axes([0.75, 0.73, 0.09, 0.02])
    self.b_g1 = Button(self.ax_g1, GAME1_SHORT)
    self.b_g1.on_clicked(g1_repeat)

    #############################
    # 4. Run tSNE on Game 2 again
    #############################

    def g2_repeat(event):
        self.update_tSNE(GAME2, SEED2)

    self.ax_g2 = plt.axes([0.85, 0.73, 0.09, 0.02])
    self.b_g2 = Button(self.ax_g2, GAME2_SHORT)
    self.b_g2.on_clicked(g2_repeat)

    #############################
    # 5. Run tSNE on Game 3 again
    #############################

    def g3_repeat(event):
        self.update_tSNE(GAME3, SEED3)

    self.ax_g3 = plt.axes([0.75, 0.70, 0.09, 0.02])
    self.b_g3 = Button(self.ax_g3, GAME3_SHORT)
    self.b_g3.on_clicked(g3_repeat)

    #############################
    # 6. Run tSNE on Game 4 again
    #############################

    def g4_repeat(event):
        self.update_tSNE(GAME4, SEED4)

    self.ax_g4 = plt.axes([0.85, 0.70, 0.09, 0.02])
    self.b_g4 = Button(self.ax_g4, GAME4_SHORT)
    self.b_g4.on_clicked(g4_repeat)

    #############################
    # 5. Perplexity Slider
    #############################

    def update_per(event):
        self.perplexity = self.b_perslider.val

    # Make a horizontal slider to control the frequency.
    self.ax_perslider = plt.axes([0.80, 0.62, 0.09, 0.02])
    self.b_perslider = Slider(
        label='Perplexity', ax=self.ax_perslider,
        valmin=30,
        valmax=150,
        valinit=60)

    self.b_perslider.on_changed(update_per)

    #############################
    # 6. SAMDP
    #############################
    def SAMDP(event):
        self.samdp()

    self.ax_samdp = plt.axes([0.75, 0.55, 0.09, 0.02])
    self.b_samdp = Button(self.ax_samdp, 'SAMDP')
    self.b_samdp.on_clicked(SAMDP)

    #############################
    # 7. Clear SAMDP
    #############################
    def clear_SAMDP(event):
        self.clear_samdp()

    self.ax_csamdp = plt.axes([0.85, 0.55, 0.09, 0.02])
    self.b_csamdp = Button(self.ax_csamdp, 'MANUAL')
    self.b_csamdp.on_clicked(clear_SAMDP)
