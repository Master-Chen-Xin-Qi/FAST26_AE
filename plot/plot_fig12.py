# -*- encoding: utf-8 -*-


import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import os
from AE_config import colors
from utils import ecdf


FIG_SIZE = (7, 5)
LINE_WIDTH = 5
FONT_SIZE = 26
MARKER_SIZE = 12
TICK_SIZE = 25
LEGEND_SIZE = 24
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)


def plot():
    user_sim_res = np.load('../data/plot_files/user_sim_res.npy', allow_pickle=True)
    public_sim_res = np.load('../data/plot_files/public_sim_res.npy', allow_pickle=True)
    user_all_res, public_all_res = [], []
    for np_array in user_sim_res:
        key, value = np_array[0], np_array[1]
        for v in value:
            user_all_res.append(v[0])
    for np_array in public_sim_res:
        key, value = np_array[0], np_array[1]
        for v in value:
            public_all_res.append(v[0])
    plt.figure(figsize=(7, 3.5))
    public_x, public_y = ecdf(public_all_res)
    plt.axvline(x=0.9, color='green', linestyle='--', linewidth=LINE_WIDTH)
    plt.plot([0, public_x[0]], [0, public_y[0]], color=colors[0], linewidth=LINE_WIDTH, drawstyle='steps-post', clip_on=False)
    plt.plot(public_x, public_y, color=colors[0], linewidth=LINE_WIDTH, label='Public', clip_on=False)
    user_x, user_y = ecdf(user_all_res)

    plt.plot([0, user_x[0]], [0, user_y[0]], color=colors[1], linewidth=LINE_WIDTH, drawstyle='steps-post', clip_on=False)
    plt.plot(user_x, user_y, color=colors[1], linewidth=LINE_WIDTH, label='User-defined', clip_on=False)
    plt.ylabel('CDF', fontsize=29)
    plt.xlabel('Similarity', fontsize=29)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 1])
    plt.ylim(-0.02, 1.01)
    plt.xlim(0, 1.01)
    plt.legend(loc='upper left')
    save_path = '../fig/fig12.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'Save figure to {os.path.abspath(save_path)}:1')

if __name__ == "__main__":
    plot()
