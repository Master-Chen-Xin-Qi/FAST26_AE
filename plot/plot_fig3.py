# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import matplotlib
import os
import polars as pl
from utils import ecdf
from AE_config import colors, meta_file, end_day

GB = 1024 * 1024 * 1024
FIG_SIZE = (7, 4)
LINE_WIDTH = 5
FONT_SIZE = 28
MARKER_SIZE = 12
TICK_SIZE = 27
LEGEND_SIZE = 26
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_vd_cdf():
    vd_size = collect_vd_size()
    plot_fig(vd_size)

def collect_vd_size():
    meta = f'../{meta_file}'
    df = pl.read_csv(meta).filter((pl.col('gmt_create').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S") <= pl.lit(end_day).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")))
    vd_size = df.select(pl.col('device_size').alias('vd_size')).to_numpy().flatten() // GB
    return vd_size

def plot_fig(vd_size):
    x, y = ecdf(vd_size)
    save_path = '../fig/fig3.pdf'
    plt.figure(figsize=FIG_SIZE)
    plt.plot(x, y, linewidth=LINE_WIDTH, color=colors[-2], label='VD size')
    plt.axvline(x=64, color='red', linestyle='--', linewidth=LINE_WIDTH)
    plt.xscale('log')
    plt.xticks([10, 100, 1000])
    plt.xlim(8, 2000)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel('CDF', fontsize=FONT_SIZE)
    plt.xlabel('VD size (GiB)', fontsize=FONT_SIZE)
    plt.ylim(0, 1.01)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f'Save cdf figure to {os.path.abspath(save_path)}:1.')


def main():
    plot_vd_cdf()

if __name__ == "__main__":
    main()
