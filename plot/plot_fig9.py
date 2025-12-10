# -*- encoding: utf-8 -*-

import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from AE_config import BLOCK_SIZE, colors

FIG_SIZE = (7, 4)
LINE_WIDTH = 5
FONT_SIZE = 28
MARKER_SIZE = 12
TICK_SIZE = 27
LEGEND_SIZE = 26
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update(parameters)

image_npys = ["../data/plot_files/image1.npy", "../data/plot_files/image2.npy"]

def plot():
    for k, image_npy in enumerate(image_npys):
        image_name = image_npy.split('/')[-1]
        if 'image2' in image_name:
            flag = True
        else:
            flag = False
        record = np.load(image_npy)
        record = record[record[:, 0] == 1]
        plt.figure(figsize=FIG_SIZE)
        start_time = record[0, 3]
        x, y = [], []
        for i in range(len(record)):
            if record[i, 3]-start_time > 2 * 60 * 1_000_000:
                break
            block = record[i, 1] // BLOCK_SIZE / 1e4
            x.append((record[i, 3] - start_time) / 1_000_000)
            y.append(block)
        plt.scatter(x, y, c=colors[-2], s=2, clip_on=False, rasterized=flag)
        plt.xticks([0, 20, 40, 60, 80, 100, 120])
        plt.xlabel('Time (s)')
        plt.ylabel('Access block\nID (1xe4)')
        plt.xlim()
        if k == 0:
            save_path = "../fig/fig9_a.pdf"
        else:
            save_path = "../fig/fig9_b.pdf"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {os.path.abspath(save_path)}:1")


def main():
    plot()

if __name__ == "__main__":
    main()
