# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from AE_config import algo_color_map, paper_algo_trans

FIG_SIZE = (12, 4.5)
LINE_WIDTH = 5
FONT_SIZE = 26
MARKER_SIZE = 12
TICK_SIZE = 26
LEGEND_SIZE = 25
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

bandwidth_1 = {'00:00:00': 0.21, '01:00:00': 0.16, '02:00:00': 0.3, '03:00:00': 0.13, '04:00:00': 0.17, '05:00:00': 0.04, '06:00:00': 0.17, '07:00:00': 0.29, '08:00:00': 0.1, '09:00:00': 0.33, '10:00:00': 0.04, '11:00:00': 0.13, '12:00:00': 0.17, '13:00:00': 0.29, '14:00:00': 0.05, '15:00:00': 0.33, '16:00:00': 0.13, '17:00:00': 0.29, '18:00:00': 0.23, '19:00:00': 0.3, '20:00:00': 0.38, '21:00:00': 0.72, "22:00:00": 0.13, "23:00:00": 0.13}
bandwidth_2 = {'00:00:00': 0.28, '01:00:00': 0.15, '02:00:00': 0.13, '03:00:00': 0.06, '04:00:00': 0.03, '05:00:00': 0.13, '06:00:00': 0.15, '07:00:00': 0.48, '08:00:00': 0.14, '09:00:00': 0.15, '10:00:00': 0.04, '11:00:00': 0.13, '12:00:00': 0.03, '13:00:00': 0.03, '14:00:00': 0.13, '15:00:00': 1.02, '16:00:00': 0.13, '17:00:00': 0.09, '18:00:00': 0.37, '19:00:00': 0.16, '20:00:00': 1.03, '21:00:00': 0.2, "22:00:00": 0.18, "23:00:00": 0.2}

for k, v in bandwidth_1.items():
    bandwidth_1[k] = v / 60 * 1024
for k, v in bandwidth_2.items():
    bandwidth_2[k] = v / 60 * 1024

def plot(args, plot_type):
    assert plot_type in ['hit_rate', 'wait_latency']
    width = 0.5
    x = np.arange(len(args.algo))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SIZE)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    handles = []
    labels = []
    for cluster_name in ['c1', 'c2']:
        results = {}
        for i, algo in enumerate(args.algo):
            base_folder = '../data/plot_files/real_world/'
            if plot_type == 'hit_rate':
                total_file = os.path.join(base_folder, f'{algo}_{cluster_name}_total_hit_rate.npy')
            elif plot_type == 'wait_latency':
                total_file = os.path.join(base_folder, f'{algo}_{cluster_name}_total_wait_latency.npy')
            total_data = np.load(total_file)
            if plot_type != 'wait_latency':
                avg_data = np.mean(total_data)
            else:
                avg_data = np.percentile(total_data, 50) / 1_000_000
            results[algo] = avg_data
            color = algo_color_map[algo]
            if algo != 'union':
                label = paper_algo_trans[algo]
            else:
                label = paper_algo_trans['union-avg']
            if cluster_name == 'c1':
                bar = ax1.bar(x[i], avg_data, color=color, label=label, width=width, edgecolor='black')
                handles.append(bar)
                labels.append(label)
            else:
                bar = ax2.bar(x[i], avg_data, color=color, label=label, width=width, edgecolor='black')
    if plot_type == 'hit_rate':
        fig.legend(handles, labels, loc='upper center', ncol=len(args.algo) // 2 +1, frameon=False, columnspacing=0.8, handletextpad=0.4, handlelength=1.2, labelspacing=0.8, bbox_to_anchor=(0.53, 1.25))
    ax1.set_xticks(x, [])
    ax1.tick_params(axis='x', bottom=False)
    edge_rate = 1
    ax1.set_xlim(left=-edge_rate * width, right=len(args.algo)-1+edge_rate*width)
    hit_text_x = 0.42
    hit_text_y = 0.8
    lat_text_x = 0.42
    lat_text_y = 0.8
    col = 'black'
    text_size = 24
    if plot_type == 'hit_rate':
        ax1.text(hit_text_x, hit_text_y, 'Cluster1', fontsize=text_size, color=col, transform=ax1.transAxes, fontweight='bold')
        ax2.text(hit_text_x, hit_text_y, 'Cluster2', fontsize=text_size, color=col, transform=ax2.transAxes, fontweight='bold')
        ax1.set_ylabel('Hit rate')
        ax2.set_ylabel('Hit rate')
        ax1.set_ylim(0, 0.5)
        ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax2.set_ylim(0, 0.3)
        ax2.set_yticks([0, 0.1, 0.2, 0.3])
    elif plot_type == 'wait_latency':
        ax1.text(lat_text_x, lat_text_y, 'Cluster1', fontsize=text_size, color=col, transform=ax1.transAxes, fontweight='bold')
        ax2.text(lat_text_x, lat_text_y, 'Cluster2', fontsize=text_size, color=col, transform=ax2.transAxes, fontweight='bold')
        ax1.set_ylabel('Lat. (ms)')
        ax2.set_ylabel('Lat. (ms)')
        ax1.set_yticks([0, 5, 10, 15, 20])
        ax2.set_yticks([0, 20, 40, 60, 80])
    else:
        assert False, "Invalid plot type"
    ax2.set_xticks(x, [])
    ax2.tick_params(axis='x', bottom=False)
    ax2.set_xlim(left=-edge_rate * width, right=len(args.algo)-1+edge_rate*width)

    save_path = f"../fig/fig28_{plot_type}.pdf"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")


def main():
    parser = argparse.ArgumentParser(description='Real world eval')
    parser.add_argument('--algo', '-a', nargs='+', type=str, default=['lazyload'], help='Algorithm name')
    args = parser.parse_args()
    plot(args, 'hit_rate')
    plot(args, 'wait_latency')

    # Example command to plot the figure:
    # python plot_fig28.py -a lazyload leap random greedy union io_count_time thinkahead

if __name__ == "__main__":
    main()
