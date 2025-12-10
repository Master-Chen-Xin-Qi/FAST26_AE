# -*- encoding: utf-8 -*-

import sys
sys.path.append("..")
from AE_config import colors
import numpy as np
import os
import matplotlib.pyplot as plt

FIG_SIZE = (8, 5)
LINE_WIDTH = 5
FONT_SIZE = 22
MARKER_SIZE = 12
TICK_SIZE = 21
LEGEND_SIZE = 20
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)


block_sizes = [4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1 * 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024]
span_rate_file = '../data/plot_files/span_io_rate.npy'


def compute_avg():
    res = np.load(span_rate_file, allow_pickle=True).item()
    block_size_avg = {"public": [], "user": []}
    for k, v in res.items():
        for i in range(len(block_sizes)):
            avg = np.mean(v[:, i])
            block_size_avg[k].append(avg)
    return block_size_avg

def plot_avg():
    if not os.path.exists('../fig'):
        os.makedirs('../fig')
    label_map = {'public': 'Public', 'user': 'User-defined'}
    block_size_avg = compute_avg()  
    plt.figure(figsize=(11, 4))
    WIDTH = 0.4
    for i, (k, v) in enumerate(block_size_avg.items()):
        if i == 0:
            x = list(range(len(block_sizes)))
        else:
            x = list(np.array(range(len(block_sizes))) + WIDTH)
        plt.bar(x, v, label=label_map[k],width=WIDTH, align='center', edgecolor='black', color=colors[i])
    plt.xticks(ticks=list(np.array(range(len(block_sizes))) + 0.5*WIDTH), labels=[f"{bs // 1024}" for bs in block_sizes])
    plt.legend(fontsize=LEGEND_SIZE+2)
    plt.xlim(-WIDTH, len(block_sizes) - 0.5*WIDTH)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=TICK_SIZE)
    plt.xlabel('Block size (KiB)', fontsize=FONT_SIZE+2)
    plt.ylabel('Cross block rate', fontsize=FONT_SIZE+2)
    save_path = "../fig/fig2.pdf"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")

def main():
    plot_avg()

if __name__ == "__main__":
    main()
