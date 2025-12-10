# -*- encoding: utf-8 -*-


import sys
sys.path.append('..')
import os
import json
import matplotlib
import matplotlib.pyplot as plt
from utils import ecdf
from AE_config import colors

FIG_SIZE = (7, 5)
LINE_WIDTH = 5
FONT_SIZE = 28
MARKER_SIZE = 12
TICK_SIZE = 24
LEGEND_SIZE = 26
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_together():
    data_file = '../data/plot_files/slow_io_rate.json'
    with open(data_file, 'r') as f:
        data = json.load(f)
    val = list(data.values())
    rate_x, rate_y = ecdf(val)

    cluster_data_file = '../data/plot_files/slow_io_cluster_rate.json'
    with open(cluster_data_file, 'r') as f:
        cluster_data = json.load(f)
    c_val = list(cluster_data.values())
    c_rate_x, crate_y = ecdf(c_val)

    plt.figure(figsize=(7, 4))
    plt.plot(rate_x, rate_y, color=colors[0], label='Lazy loading / Total')
    plt.plot(c_rate_x, crate_y, color=colors[1], label='Cluster / Total')
    plt.ylabel('CDF')
    plt.xlabel('Ratio of lazy loading per day')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.legend(loc='upper left', frameon=False, columnspacing=1, handletextpad=0.5, handlelength=1, labelspacing=0.5, ncols=1)
    save_path = "../fig/fig11.pdf"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")

if __name__ == '__main__':
    plot_together()
    