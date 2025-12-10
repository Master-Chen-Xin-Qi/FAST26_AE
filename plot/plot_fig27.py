# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from AE_config import bandwidth_colors

FIG_SIZE = (12, 3)
LINE_WIDTH = 5
FONT_SIZE = 24
MARKER_SIZE = 14
TICK_SIZE = 24
LEGEND_SIZE = 23
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

baselines = ["random", "leap", "greedy", "union-avg", "io_count_time", "thinkahead"]
trans_algo = {"random": "Random", "leap": "Leap", "greedy": "DADI+", "union-avg": "VMT-avg", "io_count_time": "IOCntT", "thinkahead": "Thinkahead"}
train_time = {'random': 0.03, 'leap': 0, 'greedy': 8.79, 'union-avg': 14.13, 'io_count_time': 26.14, 'thinkahead': 8820}
infer_time = {'random': 0.0008, 'greedy': 0.0012, 'leap': 0.003, 'union-avg': 0.0026, 'io_count_time': 0.0045, 'thinkahead': 0.0042}



def plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)
    width = 0.5
    x = np.arange(width/2, len(baselines)+width/2)
    handles = []
    labels = []
    for i, baseline in enumerate(baselines):
        train_t = train_time[baseline]
        if baseline == 'random':
            color = bandwidth_colors[3]
        elif baseline == 'greedy':
            color = bandwidth_colors[4]
        elif baseline == 'leap':
            color = bandwidth_colors[2]
        elif baseline == 'union-avg':
            color = bandwidth_colors[6]
        elif baseline == 'io_count_time':
            color = bandwidth_colors[8]
        elif baseline == 'thinkahead':
            color = bandwidth_colors[10]
        if i == len(baselines) - 1:
            bar = ax1.bar(x[i], train_t, color=color, label=f'{trans_algo[baseline]}', width=width, edgecolor='black')
            ax2.bar(x[i], infer_time[baseline] * 1_000, color=color, width=width, edgecolor='black')
            handles.append(bar)
            labels.append(baseline)
        else:
            bar = ax1.bar(x[i], train_t, color=color, label=f'{trans_algo[baseline]}', width=width, edgecolor='black')
            ax2.bar(x[i], infer_time[baseline] * 1_000, color=color, width=width, edgecolor='black')
            handles.append(bar)
            labels.append(baseline)
    ax1.set_ylabel("Overhead (s)")
    ax1.set_xticks(x, [])
    TEXT_SIZE = FONT_SIZE + 2
    ax1.text(x[0], 0.1, '0', ha='center', va='bottom', fontsize=TEXT_SIZE)
    ax1.text(x[1], 0.1, '0', ha='center', va='bottom', fontsize=TEXT_SIZE)
    ax1.text(x[-1], 40, train_time['thinkahead'], ha='center', va='bottom', fontsize=TEXT_SIZE)
    ax1.set_xlim(-width/2, len(baselines)-width/2)
    ax1.tick_params(axis='x', bottom=False)
    ax1.set_ylim(0, 50)
    ax1.set_yticks([0, 10, 20, 30, 40, 50])
    ax2.set_ylabel("Overhead (ms)")
    ax2.set_xticks(x, [])
    ax2.set_xlim(-width/2, len(baselines)-width/2)
    ax2.tick_params(axis='x', bottom=False)
    ax2.set_ylim(0, 5)
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    fig.legend(ncols=len(baselines)+1, bbox_to_anchor=(0.5, 1.15), loc='upper center', frameon=False, columnspacing=0.6, handletextpad=0.3, handlelength=1.2, labelspacing=0.6)
    plt.tight_layout(w_pad=2.5)
    save_path = "../fig/fig27.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")

if __name__ == "__main__":
    plot()
