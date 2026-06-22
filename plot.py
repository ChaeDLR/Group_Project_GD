"""
https://matplotlib.org/stable/gallery/lines_bars_and_markers/hat_graph.html
"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.offsetbox import AnchoredText


def regression_graph(ax, y_actuals, y_predictions, title=None, xlabel=None, ylabel=None, text=None):
    """plot actualy y data points and
        the model linear regression prediction points 
    """
    x = [i for i in range(len(y_actuals))]
    ax.plot(x, y_actuals, "bo", label="y actual", markevery=1, mec="1.0")
    ax.plot(x, y_predictions, "co", label="y predictions", markevery=1, mec="1.0")

    for xi, ya, yp in zip(x, y_actuals, y_predictions):
        ax.vlines(xi, min(ya, yp), max(ya, yp), color='green', alpha=0.6)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.xlabel=xlabel
    if ylabel:
        ax.ylabel=ylabel
    if text:
        anch_text = AnchoredText(
            text, prop={"size": 15}, frameon=True, loc='upper left'
        )
        anch_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(anch_text)
    
    ax.grid()
    ax.legend()

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
