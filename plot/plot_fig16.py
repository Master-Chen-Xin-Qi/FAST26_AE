# -*- encoding: utf-8 -*-

from matplotlib.patches import Rectangle
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


a = np.load('../data/plot_files/VD1.npy')
b = np.load('../data/plot_files/VD2.npy')

plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

min_len = min(len(a), len(b))
a = a[:min_len]
b = b[:min_len]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
LABEL_SIZE = 24

line1, = ax1.plot(np.array(a) / 10000, label='VD-1', color='blue', linewidth=4)
ax1.set_ylabel('Access block\nID (1xe4)', fontsize=LABEL_SIZE)
ax1.set_xlabel('Distinct accessed block', fontsize=LABEL_SIZE)

line2, = ax2.plot(np.array(b) / 10000, label='VD-2', color='orange', linewidth=4)
ax2.set_ylabel('Access block\nID (1xe4)', fontsize=LABEL_SIZE)
ax2.set_xlabel('Distinct accessed block', fontsize=LABEL_SIZE)

rec_width = 3.8
y_high = 2.0
rect1 = Rectangle((20, 0), 120, y_high, linewidth=rec_width, edgecolor='#de1f00', facecolor='none', linestyle='--', zorder=10)

rect2 = Rectangle((20, 0), 70, y_high, linewidth=rec_width, edgecolor='#de1f00', facecolor='none', linestyle='--', zorder=10)
ax1.add_patch(rect1)
ax2.add_patch(rect2)

handles = [line1, line2]
labels = [h.get_label() for h in handles]
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=24, frameon=False, bbox_to_anchor=(0.5, 1.18))
plt.tight_layout()
fig_path = '../fig/fig16.pdf'
plt.savefig(fig_path, bbox_inches='tight', dpi=300)
print(f'Save figure to {os.path.abspath(fig_path)}:1')
