# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
from utils import ecdf
from AE_config import colors
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

FIG_SIZE = (7, 5)
LINE_WIDTH = 5
FONT_SIZE = 28
MARKER_SIZE = 12
TICK_SIZE = 27
LEGEND_SIZE = 26
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_lazyload_latency_cdf():
    slow_io_file = '../data/plot_files/slow_io.npy'
    latency = np.load(slow_io_file)
    x, y = ecdf(latency)
    save_path = "../fig/fig10.pdf"
    plt.figure(figsize=(7, 3.5))
    plt.plot(x, y, color=colors[0], linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
    plt.xscale('log')
    plt.xlim(0.8, 250)
    plt.ylim(0, 1.03)
    plt.ylabel('CDF')
    plt.xlabel('Latency (s)')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    plot_lazyload_latency_cdf()
