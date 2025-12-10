# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
from AE_config import colors
from utils import ecdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

FIG_SIZE = (7, 4)
LINE_WIDTH = 5
FONT_SIZE = 28
MARKER_SIZE = 12
TICK_SIZE = 27
LEGEND_SIZE = 26
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot():
    public_res = np.load('../data/plot_files/public_interval.npy')
    user_res = np.load('../data/plot_files/user_interval.npy')
    x1, y1 = ecdf(public_res)
    x2, y2 = ecdf(user_res)

    plt.figure(figsize=FIG_SIZE)
    plt.plot(x1, y1, label='Public', color=colors[0])
    plt.plot(x2, y2, label='User-defined', color=colors[1])
    plt.xlabel('Consecutive I/O interval (us)', fontsize=FONT_SIZE+1)
    plt.ylabel('CDF', fontsize=FONT_SIZE+1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xscale('log')
    plt.xticks([0.1, 100, 100000, 100000000], [r'$10^{-1}$', r'$10^{2}$', r'$10^{5}$', r'$10^{8}$'])
    plt.legend(columnspacing=1, handletextpad=0.8, handlelength=1, labelspacing=0.8, loc='lower right', bbox_to_anchor=(1.03, -0.05))
    save_path = "../fig/fig18.pdf"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")

if __name__ == "__main__":
    plot()
