"Figure 1: median peak width per content category and era. Run: python figures.py --out figure1.png"

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from table_utils import add_category, add_common_args, analytic, load_raw

"Category: colour, marker, label. Viral is labelled at its first point, the rest at the last."
STYLE = {
    'viral':          ('#7B4F9E', 'D', 'Viral'),
    'movie':          ('#D55E00', 's', 'Movie'),
    'platform_event': ('#009E73', '^', 'Platform event'),
    'control':        ('#0072B2', 'o', 'Control'),
}
ERA_TICKS = ["2015\u20132019", "2020\u20132022", "2023\u20132026"]


def figure_1(df, out):
    df = add_category(df)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for category, (color, marker, label) in STYLE.items():
        medians = [df.loc[(df['category'] == category) & (df['era_ordinal'] == e),
                          'peak_width_days'].median() for e in (0, 1, 2)]
        ax.plot(range(3), medians, color=color, marker=marker, linewidth=2.2, markersize=6)
        if category == 'viral':
            ax.annotate(label, (0, medians[0]), xytext=(12, 4), textcoords='offset points',
                        color=color, fontsize=9, fontweight='bold', va='center')
        else:
            ax.annotate(label, (2, medians[2]), xytext=(10, 0), textcoords='offset points',
                        color=color, fontsize=9, fontweight='bold', va='center')

    ax.set_xticks(range(3))
    ax.set_xticklabels(ERA_TICKS)
    ax.set_xlim(-0.25, 2.5)
    ax.set_ylim(0, 42)
    ax.set_ylabel('Median peak width (days)')
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#888888')

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"saved {out}")


if __name__ == '__main__':
    parser = add_common_args(argparse.ArgumentParser(description="Build Figure 1."))
    parser.add_argument('--out', default='figure1.png')
    args = parser.parse_args()
    figure_1(analytic(load_raw(args.trends, args.strat, args.youtube)), args.out)
