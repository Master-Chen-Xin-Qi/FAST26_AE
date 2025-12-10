# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from AE_config import colors
from utils import ecdf

FIG_SIZE = (10, 5)
LINE_WIDTH = 5
FONT_SIZE = 30
MARKER_SIZE = 12
TICK_SIZE = 28
LEGEND_SIZE = 26
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def cache_place():
    data_file = '../data/plot_files/cluster_counts.npy'
    data = np.load(data_file)
    x, y = ecdf(data)
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, color=colors[-2])
    plt.xlabel('Cluster counts per day')
    plt.ylabel('CDF')
    plt.xticks([0, 50, 100, 150])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    save_path = '../fig/fig6.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")

def main():
    # cluster_pattern()
    cache_place()
    

if __name__ == "__main__":
    main()
