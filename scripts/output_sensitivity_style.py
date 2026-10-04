"""Shared print-scale style and colour meanings for the sensitivity paper.

Blue: reference condition. Vermillion: alternative condition. Near-black:
combined measurements and condition-neutral quantities. Grey: controls,
excluded observations, or reference guides. Marker shape separates d90/d99
and model classes; colour never encodes identifiable/unidentifiable.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

WIDTH = 6.5  # the article draft's 165.1-mm text width, included at 100%
BLUE, VERM, INK, GREY = '#0072B2', '#D55E00', '#222222', '#858585'
LIGHT, GRID = '#D9D9D9', '#E6E6E6'
D90, D99 = 'd₉₀', 'd₉₉'  # baseline font size; avoid mathtext shrinking subscripts

def log_ticks(ax, axis='y'):
    """Format log ticks at the full tick font size, including exponents."""
    target = ax.yaxis if axis == 'y' else ax.xaxis
    target.set_major_locator(LogLocator(base=10))
    def label(value, position):
        if .001 <= value <= 10000:
            return f'{value:g}'
        return f'{value:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')
    target.set_major_formatter(FuncFormatter(label))

def apply():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': .6, 'lines.linewidth': 1.2,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': INK, 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })

def panel(ax, label, title):
    ax.text(-.02, 1.10, label, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom', ha='right')
    ax.annotate(title, xy=(-.02, 1.10), xycoords='axes fraction',
                xytext=(4, 0), textcoords='offset points', va='bottom',
                fontsize=7.5, annotation_clip=False)

def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(directory / f'{name}.{extension}', dpi=600)
    plt.close(fig)
