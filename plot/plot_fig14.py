# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from AE_config import colors

FIG_SIZE = (8, 5)
LINE_WIDTH = 5
FONT_SIZE = 26
MARKER_SIZE = 12
TICK_SIZE = 25
LEGEND_SIZE = 22
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_top_image_num():
    width = 0.4
    image_num = np.load('../data/plot_files/top100_creation.npy')
    total_counts = np.sum(image_num)
    image_num = sorted(image_num, reverse=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    
    ax1.bar(range(1, len(image_num)+1), image_num, width=width, color=colors[-2], label='Count')
    ax1.set_yscale('log')
    ax1.set_yticks([1, 10, 100, 1000, 10000])
    ax1.set_ylabel('Count', fontsize=27)
    ax1.set_xlabel('Image id', fontsize=27)
    ax1.set_xlim(-1, 103)
    
    ax2 = ax1.twinx()
    cumulate_ratio = np.cumsum(image_num) / total_counts
    ax2.plot(range(1, len(image_num)+1), cumulate_ratio, color=colors[1], linewidth=LINE_WIDTH, label='Cumulation')
    ax2.set_ylabel('Cumulative ratio', fontsize=26)
    ax2.set_ylim(0, 1.05)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.33, 0.84), handletextpad=0.5, handlelength=1, labelspacing=0.5)
    
    plt.tight_layout()
    save_path = '../fig/fig14.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'Save top image num to {os.path.abspath(save_path)}:1')

if __name__ == '__main__':
    plot_top_image_num()
