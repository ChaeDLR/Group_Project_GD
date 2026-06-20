"""
https://matplotlib.org/stable/gallery/lines_bars_and_markers/hat_graph.html
"""
import matplotlib.pyplot as plt
import numpy as np


def hat_graph(ax, x_labels, values, group_labels):
    """
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x_labels : list of str
        The category names to be displayed on the x-axis.
    values : (M, N) array-like
        The data values.
        Rows are the groups (len(group_labels) == M).
        Columns are the categories (len(x_labels) == N).
    group_labels : list of str
        The labels for the groups. Displayed in the legend.
    """
    values = np.array(values)
    color_cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # draw the hats
    bars = ax.grouped_bar(
        (values - values[0]).T, bottom=values[0], tick_labels=x_labels,
        labels=group_labels, edgecolor='black', group_spacing=0.8,
        colors=['none'] + color_cycle_colors
    )

    # attach a text label on top of each bar
    for bc, heights in zip(bars.bar_containers, values):
        ax.bar_label(bc, heights, label_type='edge', padding=4)

if __name__ == "__main__":
    fig, ax = plt.subplots(layout="constrained")

    x_labels = ["I", "II", "III", "IV", "V"]
    playerA = np.asarray([5, 15, 22, 20, 25])
    playerB = np.asarray([25, 32, 34, 30, 27])

    hat_graph(ax, x_labels, [playerA, playerB], ["Player A", "Player B"])

    ax.set_title("Hat Graph Example")
    ax.legend()
    plt.show()
