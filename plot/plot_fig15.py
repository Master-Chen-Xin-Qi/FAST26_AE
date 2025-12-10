# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from utils import ecdf
from AE_config import colors

MAX_WORKERS = 40
FIG_SIZE = (6.5, 4)
LINE_WIDTH = 5
FONT_SIZE = 28
MARKER_SIZE = 12
TICK_SIZE = 27
LEGEND_SIZE = 26
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)

def plot_start_time_cdf():

    with open('../data/plot_files/public_start_time.json', 'r') as f:
        public_map = json.load(f)
    with open('../data/plot_files/user_start_time.json', 'r') as f:
        user_map = json.load(f)
    public_res, user_res = [], []
    for k, v in public_map.items():
        v = list(np.array(v) / 1_000_000)
        public_res.extend(v)
    for k, v in user_map.items():
        v = list(np.array(v) / 1_000_000)
        user_res.extend(v)
    public_x, public_y = ecdf(public_res)
    user_x, user_y = ecdf(user_res)
    plt.figure(figsize=(7, 4))
    plt.plot(public_x, public_y, color=colors[0], label='Public', linewidth=LINE_WIDTH)
    plt.plot(user_x, user_y, color=colors[1], label='User-defined', linewidth=LINE_WIDTH)
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.legend(columnspacing=1, handletextpad=0.5, handlelength=1, labelspacing=0.5)
    plt.xlabel('First I/O interval (s)', fontsize=FONT_SIZE+1)
    plt.ylabel('CDF', fontsize=FONT_SIZE+1)
    save_path = '../fig/fig15.pdf'
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'Save figure to {os.path.abspath(save_path)}:1')

if __name__ == '__main__':
    plot_start_time_cdf()