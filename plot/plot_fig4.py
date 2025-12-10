# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils import ecdf
from AE_config import colors

FIG_SIZE = (7, 4)
LINE_WIDTH = 5
FONT_SIZE = 28
MARKER_SIZE = 12
TICK_SIZE = 27
LEGEND_SIZE = 24
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot(res):
    user_res, public_res = res
    plt.figure(figsize=FIG_SIZE)
    user_x, user_y = ecdf(user_res)
    public_x, public_y = ecdf(public_res)
    plt.plot(public_x, public_y, label='Public', color=colors[0])
    plt.plot(user_x, user_y, label='User-defined', color=colors[1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlabel('Image size ratio')
    plt.ylabel('CDF')
    plt.legend(loc='lower right', columnspacing=0.8, handletextpad=0.5, handlelength=1, labelspacing=0.4, bbox_to_anchor=(1.023, -0.05))
    save_path = "../fig/fig4.pdf"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")


def main():
    public_ratio = np.load('../data/plot_files/public_image_size_ratio.npy')
    user_ratio = np.load('../data/plot_files/user_image_size_ratio.npy')
    all_res = user_ratio, public_ratio
    plot(all_res)

if __name__ == "__main__":
    main()
