# -*- encoding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import os
from AE_config import algo_color_map

FIG_SIZE = (8.5, 3)
LINE_WIDTH = 5
FONT_SIZE = 22
MARKER_SIZE = 10
TICK_SIZE = 21
LEGEND_SIZE = 20
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

without_dp = [0.9234, 4.320]
without_ss_switch = [0.9425, 4.125]
without_ss = [0.4937, 5.047]
without_zs = [0.9321, 4.253]
all = [0.9664, 3.1145]
colors = ['red', 'blue', 'green', 'yellow', algo_color_map['thinkahead']]


def plot():
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    
    ax2 = ax1.twinx()
    
    width = 0.5
    x = range(1, 6, 1)
    label0, label1, label2, label3, label4 = 'w/o DP', 'w/o BS-feat', 'w/o BS', 'w/o ZS', 'TA'
    
    bar0 = ax1.bar(x[0], without_dp[0], width=width, color=colors[0], label=label0, edgecolor='black')
    bar1 = ax1.bar(x[1], without_ss_switch[0], width=width, color=colors[1], label=label1, edgecolor='black')
    bar2 = ax1.bar(x[2], without_ss[0], width=width, color=colors[2], label=label2, edgecolor='black')
    bar3 = ax1.bar(x[3], without_zs[0], width=width, color=colors[3], label=label3, edgecolor='black')
    bar4 = ax1.bar(x[4], all[0], width=width, color=colors[4], label=label4, edgecolor='black')
    
    latency_data = [without_dp[1], without_ss_switch[1], without_ss[1], without_zs[1], all[1]]
    line, = ax2.plot(x, latency_data, 'o-', color='black', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, zorder=3)
    
    ax1.set_ylabel('Hit rate', color='black')
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.5, 1])
    ax1.tick_params(axis='y', labelcolor='black')
    
    ax2.set_ylabel('P99 Lat. (s)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([])
    ax2.set_ylim(0, 5.5)
    ax2.set_yticks([0, 2.5, 5])
    ax1.tick_params(axis='x', bottom=False)
    
    handles = [bar0, bar1, bar2, bar3, bar4, line]
    labels = [label0, label1, label2, label3, label4]
    ax1.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, 
               columnspacing=0.8, handletextpad=0.4, handlelength=1.2, 
               labelspacing=0.6, bbox_to_anchor=(0.5, 1.4))
    
    plt.tight_layout()
    save_path = "../fig/fig29.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")

if __name__ == "__main__":
    plot()
