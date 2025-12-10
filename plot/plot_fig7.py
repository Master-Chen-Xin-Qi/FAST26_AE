# -*- encoding: utf-8 -*-


import sys
sys.path.append('..')
import matplotlib
import matplotlib.pyplot as plt
import os
import json
from AE_config import colors

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


def plot():
    plt.figure(figsize=FIG_SIZE)
    save_path = ""
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")


def cluster_pattern():

    file_path = '../data/plot_files/cluster_image.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    plt.figure(figsize=(8, 4))
    for i in range(len(data)):
        cluster_data = data[f'{i+1}']
        time = range(len(cluster_data))
        plt.plot(time, cluster_data, label=f'C{i+1}', color=colors[i])
    for i in range(9, 168, 24):
        plt.axvline(i, color=colors[-2], linestyle='--', linewidth=3)
    plt.legend(ncol=5, loc='upper left', frameon=False, columnspacing=0.6, handletextpad=0.4, handlelength=1, labelspacing=0.7, bbox_to_anchor=(-0.025, 1.21))
    plt.ylabel('Image count')
    plt.yticks([0, 100, 200, 300])
    plt.xticks(range(9, 168, 24), ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'])
    plt.ylim(0, 300)
    save_path = '../fig/fig7.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")


    
if __name__ == "__main__":
    cluster_pattern()
