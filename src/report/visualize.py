import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Text


def item_hist(df: pd.DataFrame):
    counts_of_items = df['item_key'].value_counts()
    plot_df = pd.DataFrame({'item': counts_of_items.index.to_list(), 'counts': counts_of_items.to_list()})
    plot_df = plot_df[plot_df.counts > 1200]

    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white', dpi=80)
    ax.vlines(x=plot_df.index, ymin=0, ymax=plot_df.counts, color='firebrick', alpha=0.7, linewidth=20)

    for i, counts in enumerate(plot_df.counts):
        ax.text(i, counts + 50, round(counts, 1), horizontalalignment='center')

    ax.set_title('Гистограмма кол-ва сессий с товаром', fontdict={'size': 22})
    ax.set(ylabel='Кол-во сессий', ylim=(0, 3000))
    plt.xticks(plot_df.index, plot_df.item, rotation=60, horizontalalignment='right', fontsize=12)

    return plt.gcf()


def session_hist(df: pd.DataFrame):
    counts_of_session = df['session_key'].value_counts()
    plot_df = pd.DataFrame({'session': counts_of_session.index.to_list(), 'counts': counts_of_session.to_list()})
    plot_df = plot_df[plot_df.counts > 600]

    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white', dpi=80)
    ax.vlines(x=plot_df.index, ymin=0, ymax=plot_df.counts, color='firebrick', alpha=0.7, linewidth=20)

    for i, counts in enumerate(plot_df.counts):
        ax.text(i, counts + 50, round(counts, 1), horizontalalignment='center')

    ax.set_title('Гистограмма кол-ва товаров в сессии', fontdict={'size': 22})
    ax.set(ylabel='Кол-во товаров', ylim=(0, 1200))
    plt.xticks(plot_df.index, plot_df.session, rotation=60, horizontalalignment='right', fontsize=12)

    return plt.gcf()

