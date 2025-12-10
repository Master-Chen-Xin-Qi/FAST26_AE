# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib
from utils import ecdf
from AE_config import colors, meta_file

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


def plot_user_image_num(user_image_num):
    plt.figure(figsize=(7, 4))
    x, y = ecdf(user_image_num)
    x = np.insert(x, 0, 1)
    y = np.insert(y, 0, 0)
    plt.plot(x, y, color=colors[-2], linewidth=LINE_WIDTH)
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000, 10000, 100000])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylim(0, 1.02)
    plt.ylabel('CDF')
    plt.xlabel('Images accessed by a user', horizontalalignment='left', x=0.04)
    save_path = '../fig/fig5.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'Save cdf to {os.path.abspath(save_path)}:1')

if __name__ == "__main__":
    df = pl.read_csv(f'../{meta_file}')
    user_image_num = df.group_by('user_id').agg(pl.count('image_id').alias('image_num')).sort('image_num', descending=True)['image_num'].to_numpy()
    plot_user_image_num(user_image_num)

