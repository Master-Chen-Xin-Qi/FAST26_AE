# -*- encoding: utf-8 -*-

# -*- encoding: utf-8 -*-
'''
@File      : plot_cluster_bw.py
@Describe  : 画出几个集群小时级别的带宽变化趋势
@Time      : 2025/06/04 11:54:26
@Author    : xinqichen
'''

import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import matplotlib
import json
import os
from AE_config import colors

FIG_SIZE = (9, 4)
LINE_WIDTH = 5
FONT_SIZE = 25
MARKER_SIZE = 12
TICK_SIZE = 25
LEGEND_SIZE = 26
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot():
    bandwidth_file = '../data/plot_files/cluster_bandwidth.json'
    with open(bandwidth_file, 'r') as f:
        bandwidth = json.load(f)
    plt.figure(figsize=FIG_SIZE)
    for i in range(len(bandwidth)):
        cluster_bw = bandwidth[f'{i}']
        plt.plot(cluster_bw[:100], label=f'C{i+1}', color=colors[i])
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.04), ncols=5, columnspacing=1.2, handletextpad=0.8, handlelength=0.8, labelspacing=0.8, frameon=False)
    plt.xlabel('Time (hour)', fontsize=29)
    plt.ylabel('Bandwidth (MB/s)', fontsize=29)
    save_path = "../fig/fig17.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")


def main():
    plot()

if __name__ == "__main__":
    main()
