"""Reproducible publication figures from validated C-MAPSS CSV tables.

Run from either the project root or scripturi/:
    python scripturi/generate_mdpi_figures.py
    python generate_mdpi_figures.py

Reads rezultate/tabele/table_{2,8,9,12}.csv.
Writes PNG (300 dpi), PDF and SVG to rezultate/figuri_mdpi/.
Does not modify the source CSVs.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == 'scripturi' else SCRIPT_DIR
SOURCE = ROOT / 'rezultate' / 'tabele'
DEST = ROOT / 'rezultate' / 'figuri_mdpi'
DEST.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
})

MODELS = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'LSTM', 'RF-LSTM']
SHORT = {'Logistic Regression': 'LR', 'Random Forest': 'RF',
         'Gradient Boosting': 'GB', 'LSTM': 'LSTM', 'RF-LSTM': 'RF-LSTM'}
# Colorblind-aware, distinct colors; orange is used for RF.
COLORS = {
    'Logistic Regression': '#0072B2',  # blue
    'Random Forest': '#E69F00',        # orange
    'Gradient Boosting': '#009E73',    # green
    'LSTM': '#CC79A7',                 # purple
    'RF-LSTM': '#D55E00',              # vermilion
}
MARKERS = {'Logistic Regression': 'o', 'Random Forest': 's',
           'Gradient Boosting': '^', 'LSTM': 'D', 'RF-LSTM': 'v'}
LINESTYLES = {'Logistic Regression': '-', 'Random Forest': '--',
              'Gradient Boosting': '-.', 'LSTM': ':', 'RF-LSTM': (0, (5, 2, 1, 2))}
METRIC_RE = re.compile(r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)\s*$')


def load(n):
    path = SOURCE / f'table_{n}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Check the project folder and CSV generation.')
    return pd.read_csv(path)


def mean_sd(value):
    match = METRIC_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f'Expected mean (SD), received {value!r}')
    return float(match.group(1)), float(match.group(2))


def save(fig, name):
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(DEST / f'{name}.{ext}', bbox_inches='tight', dpi=300,
                    facecolor='white')
    plt.close(fig)
    print(f'Generated: {name} (PNG/PDF/SVG)')


def grid(ax):
    ax.grid(axis='y', color='#D9DEE3', linewidth=.7, alpha=.7)
    ax.set_axisbelow(True)


# Figure A — Table 2: paired benchmark metrics.
df = load(2).set_index('model').loc[MODELS]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(MODELS))
width = .35
for j, (metric, color) in enumerate([
    ('PME utility', '#0072B2'), ('AUPRC', '#E69F00')
]):
    values = np.array([mean_sd(v) for v in df[metric]])
    ax.bar(x + (j - .5) * width, values[:, 0], width,
           yerr=values[:, 1], capsize=3, label=metric,
           color=color, edgecolor='white', linewidth=.6,
           error_kw={'elinewidth': .9, 'ecolor': '#333333'})
ax.set_xticks(x, [SHORT[m] for m in MODELS])
ax.set_ylabel('Score (mean ± SD)')
ax.set_ylim(0, 1.06)
ax.legend(frameon=False, ncol=2, loc='upper right')