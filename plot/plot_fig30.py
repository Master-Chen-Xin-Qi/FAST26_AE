# -*- encoding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from AE_config import algo_color_map

FIG_SIZE = (9, 4)
LINE_WIDTH = 5
FONT_SIZE = 22
MARKER_SIZE = 12
TICK_SIZE = 21
LEGEND_SIZE = 20
parameters = {'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE, 'legend.fontsize': LEGEND_SIZE, 'axes.labelsize': FONT_SIZE, 'figure.titlesize': FONT_SIZE, 'lines.linewidth': LINE_WIDTH, 'lines.markersize': MARKER_SIZE}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

lazyload_res = [1, 1, 1, 1]
thinkahead_res = [0.3125, 0.74, 0.7957, 0.186868]

def plot():
    x = np.arange(4)
    width = 0.35
    plt.figure(figsize=FIG_SIZE)
    save_path = "../fig/fig30.pdf"
    plt.xticks(x, ['P50 latency', 'P99 latency', 'Max latency', 'Slow I/O'])
    lazyload_bars = plt.bar(x-width/2, lazyload_res, color=algo_color_map['lazyload'], label='Lazyload', width=width, edgecolor='black')
    thinkahead_bars = plt.bar(x+width/2, thinkahead_res, color=algo_color_map['thinkahead'], label='ThinkAhead', width=width, edgecolor='black')
    plt.ylabel('Relative performance')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.legend(loc='upper center', ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.4, handlelength=1.2, labelspacing=0.8, bbox_to_anchor=(0.5, 1.2))
    lazyload_text = ['204 ms', '269 ms', '279 ms', '396']
    lazyload_text = [t.replace(' ', '\u2009') for t in lazyload_text]
    for i, bar in enumerate(lazyload_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{lazyload_text[i]}',
                ha='center', va='bottom', 
                fontsize=16, fontweight='bold')
    for i in range(len(thinkahead_bars)):
        improvement = lazyload_res[i] / thinkahead_res[i] 
        
        start_x = thinkahead_bars[i].get_x() + thinkahead_bars[i].get_width()/2
        start_y = thinkahead_bars[i].get_height()-0.01
        
        end_x = thinkahead_bars[i].get_x() + thinkahead_bars[i].get_width()/2
        end_y = lazyload_bars[i].get_height()
        
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        plt.annotate('',
                     xy=(end_x, end_y),
                     xytext=(start_x, start_y), 
                     arrowprops=dict(arrowstyle='->,head_width=0.3,head_length=0.3', 
                                     color='black', lw=2),
                     ha='center', va='center')
        
        plt.text(mid_x + 0.03, mid_y,
                 f'{improvement:.2f}x', 
                 ha='left', va='center', 
                 fontsize=16)
    plt.ylim(0, 1.13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(save_path)}:1")


if __name__ == "__main__":
    plot()
    