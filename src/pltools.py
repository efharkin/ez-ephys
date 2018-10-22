#%% IMPORT MODULES

import matplotlib.pyplot as plt
import numpy as np

#%% DEFINE HANDY TOOLS

# Function to make a set of scalebars for a mpl plot
def add_scalebar(x_units = None, y_units = None, anchor = (0.98, 0.02),
x_size = None, y_size = None, y_label_space = 0.02, x_label_space = -0.02,
bar_space = 0.06, x_on_left = True, linewidth = 3, remove_frame = True,
omit_x = False, omit_y = False, round = True, usetex = True, ax = None):

    """
    Automagically add a set of x and y scalebars to a matplotlib plot

    Inputs:

        x_units: str or None

        y_units: str or None

        anchor: tuple of floats
        --  bottom right of the bbox (in axis coordinates)

        x_size: float or None
        --  Manually set size of x scalebar (or None for automatic sizing)

        y_size: float or None
        --  Manually set size of y scalebar (or None for automatic sizing)

        text_spacing: tuple of floats
        --  amount to offset labels from respective scalebars (in axis units)

        bar_space: float
        --  amount to separate bars from eachother (in axis units)

        linewidth: numeric
        --  thickness of the scalebars

        remove_frame: bool (default False)
        --  remove the bounding box, axis ticks, etc.

        omit_x/omit_y: bool (default False)
        --  skip drawing the x/y scalebar

        round: bool (default True)
        --  round units to the nearest integer

        ax: matplotlib.axes object
        --  manually specify the axes object to which the scalebar should be added
    """

    # Basic input processing.

    if ax is None:
        ax = plt.gca()

    if x_units is None:
        x_units = ''
    if y_units is None:
        y_units = ''

    # Do y scalebar.
    if not omit_y:

        if y_size is None:
            y_span = ax.get_yticks()[:2]
            y_length = y_span[1] - y_span[0]
            y_span_ax = ax.transLimits.transform(np.array([[0, 0], y_span]).T)[:, 1]
        else:
            y_length = y_size
            y_span_ax = ax.transLimits.transform(np.array([[0, 0], [0, y_size]]))[:, 1]
        y_length_ax = y_span_ax[1] - y_span_ax[0]

        if round:
            y_length = int(np.round(y_length))

        # y-scalebar label

        if y_label_space <= 0:
            horizontalalignment = 'left'
        else:
            horizontalalignment = 'right'

        if usetex:
            y_label_text = '${}${}'.format(y_length, y_units)
        else:
            y_label_text = '{}{}'.format(y_length, y_units)

        ax.text(
        anchor[0] - y_label_space, anchor[1] + y_length_ax / 2 + bar_space,
        y_label_text,
        verticalalignment = 'center', horizontalalignment = horizontalalignment,
        size = 'small', transform = ax.transAxes
        )

        # y scalebar
        ax.plot(
        [anchor[0], anchor[0]],
        [anchor[1] + bar_space, anchor[1] + y_length_ax + bar_space],
        'k-', linewidth = linewidth,
        clip_on = False, transform = ax.transAxes
        )

    # Do x scalebar.
    if not omit_x:

        if x_size is None:
            x_span = ax.get_xticks()[:2]
            x_length = x_span[1] - x_span[0]
            x_span_ax = ax.transLimits.transform(np.array([x_span, [0, 0]]).T)[:, 0]
        else:
            x_length = x_size
            x_span_ax = ax.transLimits.transform(np.array([[0, 0], [x_size, 0]]))[:, 0]
        x_length_ax = x_span_ax[1] - x_span_ax[0]

        if round:
            x_length = int(np.round(x_length))

        # x-scalebar label
        if x_label_space <= 0:
            verticalalignment = 'top'
        else:
            verticalalignment = 'bottom'

        if x_on_left:
            Xx_text_coord = anchor[0] - x_length_ax / 2 - bar_space
            Xx_bar_coords = [anchor[0] - x_length_ax - bar_space, anchor[0] - bar_space]
        else:
            Xx_text_coord = anchor[0] + x_length_ax / 2 + bar_space
            Xx_bar_coords = [anchor[0] + x_length_ax + bar_space, anchor[0] + bar_space]

        if usetex:
            x_label_text = '${}${}'.format(x_length, x_units)
        else:
            x_label_text = '{}{}'.format(x_length, x_units)

        ax.text(
        Xx_text_coord, anchor[1] + x_label_space,
        x_label_text,
        verticalalignment = verticalalignment, horizontalalignment = 'center',
        size = 'small', transform = ax.transAxes
        )

        # x scalebar
        ax.plot(
        Xx_bar_coords,
        [anchor[1], anchor[1]],
        'k-', linewidth = linewidth,
        clip_on = False, transform = ax.transAxes
        )

    if remove_frame:
        ax.axis('off')


def hide_ticks(ax = None):

    """
    Delete the x and y ticks of the specified axes. If no axes object is provided, defaults to the current axes.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])


def hide_border(sides = 'a', ax = None):

    """
    Sides should be set to `a` for all, or a string containing `r/l/t/b` as needed.
    """

    # Check for correct input
    if not any([letter in sides for letter in 'arltb']):
        raise ValueError('sides should be passed a string with `a` for all sides, or r/l/t/b as-needed for other sides.')

    if ax is None:
        ax = plt.gca()

    if sides == 'a':
        sides = 'rltb'

    sidekeys = {
    'r': 'right',
    'l': 'left',
    't': 'top',
    'b': 'bottom'
    }

    for key, side in sidekeys.iteritems():

        if key not in sides:
            continue
        else:
            ax.spines[side].set_visible(False)


def p_to_string(p):

    """
    Takes a p-value and converts it to a pretty LaTeX string.

    p is presented to three decimal places if p >= 0.05, and as p < 0.05/0.01/0.001 otherwise.
    """

    p_rounded = np.round(p, 3)

    if p_rounded >= 0.05:
        p_str = '$p = {}$'.format(p_rounded)
    elif p_rounded < 0.05 and p_rounded >= 0.01:
        p_str = '$p < 0.05$'
    elif p_rounded < 0.01 and p_rounded >= 0.001:
        p_str = '$p < 0.01$'
    else:
        p_str = '$p < 0.001$'

    return p_str
