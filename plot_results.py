import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
# disable interactive display (do not open window)
plt.ioff()
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH   = 'results/intrinsic/intrinsic_effect_size_table.csv'
OUTPUT_DIR = 'results/intrinsic/'
# ─────────────────────────────────────────────────────────────────────────────

SHORTCUT_COLORS = {
    'sentiment':  '#ef537d',
    'perplexity': '#2f7f6f',
}
DATASET_ORDER = ['COCO', 'gossipcop', 'kaggle1', 'kaggle2', 'pheme']

def darken(c, f=0.55):
    r, g, b, _ = mcolors.to_rgba(c)
    return (r*f, g*f, b*f, 1.0)

def lighten(c, f=0.5):
    r, g, b, _ = mcolors.to_rgba(c)
    return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f, 1.0)

def main():
    df = pd.read_csv(CSV_PATH)

    # Average delta across models (BERT + DeBERTa) per dataset × shortcut
    agg = (df.groupby(['dataset', 'shortcut'])[['delta_accuracy', 'delta_macro_f1']]
             .mean()
             .reset_index())

    shortcuts = ['sentiment', 'perplexity']
    datasets  = [d for d in DATASET_ORDER if d in agg['dataset'].unique()]
    n_ds      = len(datasets)

    bar_h     = 0.30
    group_gap = 0.15
    y_centers = np.arange(n_ds) * (len(shortcuts) * bar_h + group_gap)

    # ── figure: single ACM column ──────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(4.2, 3.5),          # slightly wider to reduce crowding
        sharey=True,
    )
    # enforce exact figure size (inches)
    fig.set_size_inches(4.2, 3.5, forward=True)
    fig.subplots_adjust(wspace=0.22, bottom=0.25)

    metrics = [
        ('delta_accuracy', 'Accuracy Drop', (-0.6, 0)),
        ('delta_macro_f1', 'Macro-F1 Drop', (-0.4, 0)),
    ]

    for ax, (metric, xlabel, xlim) in zip(axes, metrics):
        for s_idx, shortcut in enumerate(shortcuts):
            sub    = agg[agg['shortcut'] == shortcut].set_index('dataset')
            values = [sub.loc[d, metric] if d in sub.index else 0.0 for d in datasets]
            y_pos  = y_centers + s_idx * bar_h

            base  = SHORTCUT_COLORS[shortcut]
            fill  = lighten(base, 0.45)
            edge  = darken(base, 0.60)

            bars = ax.barh(
                y_pos, values,
                height=bar_h,
                color=fill,
                edgecolor=edge,
                linewidth=1.4,
                zorder=3,
            )

            # value labels inside bars
            for bar, val in zip(bars, values):
                if abs(val) < 0.01:
                    continue
                label_x = val / 2
                ax.text(
                    label_x,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:+.2f}',
                    ha='center', va='center',
                    fontsize=5.5, fontweight='bold',
                    color=edge, rotation=0,
                    zorder=4,
                )

        ax.axvline(0, color='black', linewidth=0.8, zorder=2)
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel, fontsize=7, fontweight='bold')
        ax.tick_params(axis='x', labelsize=6.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    # y-axis labels on the left panel only
    tick_pos = y_centers + bar_h * (len(shortcuts) - 1) / 2
    axes[0].set_yticks(tick_pos)
    axes[0].set_yticklabels(datasets, fontsize=7)

    # shared legend inside the figure
    handles = [
        mpatches.Patch(facecolor=lighten(SHORTCUT_COLORS[s], 0.45),
                       edgecolor=darken(SHORTCUT_COLORS[s], 0.60),
                       linewidth=1.4, label=s.capitalize())
        for s in shortcuts
    ]
    fig.legend(
        handles=handles,
        fontsize=6.5,
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.08),
        framealpha=0.95,
    )

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    
    # Save as PDF
    save_path_pdf = out / 'intrinsic_diverging_delta_paper.pdf'
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
    print(f'✓ Saved: {save_path_pdf}')
    
    # Save as PNG
    save_path_png = out / 'intrinsic_diverging_delta_paper.png'
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f'✓ Saved: {save_path_png}')
    
    # close figure instead of showing it
    plt.close(fig)

if __name__ == '__main__':
    main()