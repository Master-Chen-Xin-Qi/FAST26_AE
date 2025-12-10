# -*- encoding: utf-8 -*-

import sys
sys.path.append('..')
from AE_config import bandwidth_colors, paper_algo_trans
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

FIG_SIZE = (24, 3.5)
LINE_WIDTH = 5
FONT_SIZE = 22
MARKER_SIZE = 12
TICK_SIZE = 22
LEGEND_SIZE = 20
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

SMALL_WIDTH = 0.75
BIG_WIDTH = 1.2

def get_paper_algo_trans(key):
    if key.startswith("io_count_two_time_"):
        return "IOCnt2T"
    elif key.startswith("io_count_time_"):
        return "IOCntT"
    elif key.startswith("zero_"):
        return "ThinkAhead"
    return paper_algo_trans[key]


def plot_bar(args):
    plot_hit_rate(args)
    plot_wait_latency(args)

def plot_hit_rate(args):
    if len(args.algo) >= 6:
        plt.figure(figsize=FIG_SIZE)
        width = SMALL_WIDTH
    else:
        plt.figure(figsize=FIG_SIZE)
        width = BIG_WIDTH
    all_algos = ""
    for algo in args.algo:
        all_algos += get_paper_algo_trans(algo) + '-'
    all_algos = all_algos[:-1]
    all_algos += '-ub'
    for i, bw in enumerate(args.bandwidth):
        ub_file = f'../data/plot_files/{args.mode}/oracle_hit_rate_{bw}MB.npy'
        ub_data = np.load(ub_file)
        ub_avg = np.mean(ub_data)
        algo_hit_rates = []
        for algo in args.algo:
            algo_file = f'../data/plot_files/{args.mode}/{algo}_hit_rate_{bw}MB.npy'
            algo_hit_rate = np.load(algo_file)
            algo_hit_rates.append(algo_hit_rate)
        if i == len(args.bandwidth) - 1:
            for j in range(len(args.algo)):
                algo_avg = np.mean(algo_hit_rates[j])
                label = get_paper_algo_trans(args.algo[j])
                plt.bar(i * len(args.algo) + j * width, algo_avg, width=width, color=bandwidth_colors[j+1], label=label, edgecolor='black')
            plt.bar(i * len(args.algo) + len(args.algo) * width, ub_avg, width=width, color=bandwidth_colors[0], label='Oracle', edgecolor='black')
        else:
            for j in range(len(args.algo)):
                algo_avg = np.mean(algo_hit_rates[j])
                label = get_paper_algo_trans(args.algo[j])
                plt.bar(i * len(args.algo) + j * width, algo_avg, width=width, color=bandwidth_colors[j+1], edgecolor='black')
            plt.bar(i * len(args.algo) + len(args.algo) * width, ub_avg, width=width, color=bandwidth_colors[0], edgecolor='black')
    plt.xticks(np.arange(0, len(args.bandwidth) * len(args.algo), len(args.algo)) + width / 2 * len(args.algo), [f'{bw}' for bw in args.bandwidth])
    total_width = len(args.bandwidth) * (len(args.algo))
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlim(left=-2 * width, right=total_width-0.5*width)
    plt.xlabel('Bandwidth (MB/s)')
    plt.ylabel('Hit rate')
    plt.legend(ncols=len(args.algo)+1, bbox_to_anchor=(0.5, 1.28), loc='upper center', frameon=False, columnspacing=1.15, handletextpad=0.4, handlelength=1.5, labelspacing=1)
    if args.mode == 'zero_shot':
        save_path = '../fig/fig26_hit_rate.pdf'
    else:
        save_path = '../fig/fig25_hit_rate.pdf'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")

def plot_wait_latency(args):
    latency_file = f'../data/plot_files/{args.mode}/p50_wait_latency.npy'
    latency = np.load(latency_file, allow_pickle=True).item()
    if len(args.algo) >= 6:
        plt.figure(figsize=FIG_SIZE)
        width = SMALL_WIDTH
    else:
        plt.figure(figsize=FIG_SIZE)
        width = BIG_WIDTH
    all_algos = ""
    for algo in args.algo:
        all_algos += get_paper_algo_trans(algo) + '-'
    all_algos = all_algos[:-1]
    all_algos += '-ub'
    ax = plt.gca()
    for i, bw in enumerate(args.bandwidth):
        bw = str(bw)
        ub_p50 = latency[bw]['oracle']
        if i == len(args.bandwidth) - 1:
            for j in range(len(args.algo)):
                algo_p50 = latency[bw][args.algo[j]]
                label = get_paper_algo_trans(args.algo[j])
                plt.bar(i * len(args.algo) + j * width, algo_p50, width=width, color=bandwidth_colors[j+1], label=label, edgecolor='black')
            plt.bar(i * len(args.algo) + len(args.algo) * width, ub_p50, width=width, color=bandwidth_colors[0], label='Oracle', edgecolor='black')
        else:
            for j in range(len(args.algo)):
                algo_p50 = latency[bw][args.algo[j]]
                label = get_paper_algo_trans(args.algo[j])
                plt.bar(i * len(args.algo) + j * width, algo_p50, width=width, color=bandwidth_colors[j+1], edgecolor='black')
            plt.bar(i * len(args.algo) + len(args.algo) * width, ub_p50, width=width, color=bandwidth_colors[0], edgecolor='black')
    plt.xticks(np.arange(0, len(args.bandwidth) * len(args.algo), len(args.algo)) + width / 2 * len(args.algo), [f'{bw}' for bw in args.bandwidth])
    total_width = len(args.bandwidth) * (len(args.algo))
    plt.yscale('log')
    plt.yticks([10000,1000000,100000000], [r'$10^4$', r'$10^6$', r'$10^8$'])
    plt.xlim(left=-2 * width, right=total_width)
    plt.xlabel('Bandwidth (MB/s)')
    plt.ylabel('P50 wait lat. (us)')
    TEXT_SIZE = FONT_SIZE+6
    y_rate = 0.03
    if args.mode == 'few_shot':  
        plt.text(0.267, y_rate, '0', fontsize=TEXT_SIZE, color='black', transform=ax.transAxes)
        plt.text(0.36, y_rate, '0', fontsize=TEXT_SIZE, color='black', transform=ax.transAxes)
        x_min = -1 * width
        x_max = total_width
        positions = [43.75, 53.75, 63.75, 73.75, 83.75, 93.75]
        for pos in positions:
            relative_x = (pos - x_min) / (x_max - x_min)
            if pos == 93.75:
                relative_x -= 0.0043
            elif pos == 83.75:
                relative_x -= 0.0034
            elif pos == 73.75:
                relative_x -= 0.003
            elif pos == 63.75:
                relative_x -= 0.0022
            elif pos == 53.75:
                relative_x -= 0.0014
            plt.text(relative_x, 0.02, '0', fontsize=TEXT_SIZE, color='black', transform=ax.transAxes)
    else:
        plt.text(0.473, y_rate, '0', fontsize=TEXT_SIZE, color='black', transform=ax.transAxes)
        plt.text(0.55, y_rate, '0', fontsize=TEXT_SIZE, color='black', transform=ax.transAxes)
        plt.text(0.573, y_rate, '0', fontsize=TEXT_SIZE, color='black', transform=ax.transAxes)
        plt.text(0.625, y_rate, '0', fontsize=TEXT_SIZE, color='black', transform=ax.transAxes)
        plt.text(0.648, y_rate, '0', fontsize=TEXT_SIZE, color='black', transform=ax.transAxes)
        x_min = -1 * width
        x_max = total_width
        positions = [73.75, 83.75, 93.75]
        for pos in positions:
            relative_x = (pos - x_min) / (x_max - x_min)
            if pos == 93.75:
                relative_x -= 0.0043
            elif pos == 83.75:
                relative_x -= 0.0034
            elif pos == 73.75:
                relative_x -= 0.003
            elif pos == 63.75:
                relative_x -= 0.0022
            elif pos == 53.75:
                relative_x -= 0.0014
            plt.text(relative_x, 0.02, '0', fontsize=TEXT_SIZE, color='black', transform=ax.transAxes)
    if args.mode == 'zero_shot':
        save_path = "../fig/fig26_p50_wait_latency.pdf"
    else:
        save_path = "../fig/fig25_p50_wait_latency.pdf"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")


def main():
    parser = argparse.ArgumentParser(description='Plot bandwidth comparison')
    parser.add_argument('--algo', '-a', nargs='+', type=str, required=True, help='Algorithm name')
    parser.add_argument('--mode', '-m', default='few_shot', choices=['few_shot', 'zero_shot'], help='Mode')
    parser.add_argument('--bandwidth', '-bw', type=int, nargs='+', required=True, help='Bandwidth in MB')
    args = parser.parse_args()
    plot_bar(args)

    # For sufficient data, run: python3 plot_fig25_26.py -a lazyload leap random greedy union-min union-avg topn io_count_time_0.5 io_count_two_time_0.5_0.3 thinkahead -bw 2 4 6 8 10 30 50 70 90 150 -m few_shot
    # For zero-shot, run: python3 plot_fig25_26.py -a lazyload leap random greedy union-min union-avg topn io_count_time_0.5 io_count_two_time_0.1_0.2 thinkahead -bw 2 4 6 8 10 30 50 70 90 150 -m zero_shot


if __name__ == "__main__":
    main()
    # file1 = '../data/plot_files/few_shot/p50_wait_latency.npy'
    # data = np.load(file1, allow_pickle=True).item()
    # for k, v in data.items():
    #     for k2, v2 in v.items():
    #         if k2 == 'genetic_score':
    #             vv = v2
    #     del v['genetic_score']
    #     v['thinkahead'] = vv
    # print(len(data))
    # np.save('../data/plot_files/few_shot/p50_wait_latency.npy', data)

    # file2 = '../data/plot_files/zero_shot/p50_wait_latency.npy'
    # data = np.load(file2, allow_pickle=True).item()
    # for k, v in data.items():
    #     for k2, v2 in v.items():
    #         if k2 == 'zero_0.1_0.2':
    #             vv = v2
    #     del v['zero_0.1_0.2']
    #     v['thinkahead'] = vv
    # np.save('../data/plot_files/zero_shot/p50_wait_latency.npy', data)
