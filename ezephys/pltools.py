# IMPORT MODULES

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# DEFINE HANDY TOOLS

# Function to make a set of scalebars for a mpl plot


def add_scalebar(x_units=None, y_units=None, anchor=(0.98, 0.02),
                 x_size=None, y_size=None, y_label_space=0.02, x_label_space=-0.02,
                 bar_space=0.06, x_on_left=True, linewidth=3, remove_frame=True,
                 omit_x=False, omit_y=False, round=True, usetex=True, ax=None):
    """Automagically add a set of x and y scalebars to a matplotlib plot.

    By default, scalebars are sized automatically based on x and y tick
    spacing.

    Arguments
    ---------
    x_units, y_units: str or None
        Units (e.g. mV, pA, ms) for x and y scalebars.
    anchor: pair of floats
        Bottom right of scale bar bbox (in axis coordinates).
    x_size, y_size: float or None
        Manually set size of x and y scalebars (or None for automatic sizing).
    x_label_space, y_label_space: float
        Offset for x and y scalebar labels (in axis units).
    bar_space: float >= 0.0
        Spacing between x and y scalebars (in axis units). Set to zero for the
        two bars to be joined at the corner.
    x_on_left: bool, default True
        Place x scalebar on left of y scalebar. Set to False to place on right
        instead.
    linewidth: float, default 3
        Thickness of scalebars.
    remove_frame: bool, default True
        Remove the frame, ticks, etc. from the axis object to which the
        scalebar is being added. True by default because the scalebars are
        assumed to be a replacement for the axes.
    omit_x, omit_y: bool, default False
        Skip drawing x or y scalebar.f
    round: bool, default True
        Round values to the nearest integer. Causes `100.0 pA` to be printed as
        `100 pA`.
    usetex: bool, default True
        Print numbers in LaTeX math mode. Set to False if LaTeX is not being
        used to render text, or `100 pA` will print as `$100$ pA`.
    ax: matplotlib.axes object, defaults to current axes
        Axis to which scalebars should be added.

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
            y_span_ax = ax.transLimits.transform(
                np.array([[0, 0], y_span]).T)[:, 1]
        else:
            y_length = y_size
            y_span_ax = ax.transLimits.transform(
                np.array([[0, 0], [0, y_size]]))[:, 1]
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
            verticalalignment='center', horizontalalignment=horizontalalignment,
            size='small', transform=ax.transAxes
        )

        # y scalebar
        ax.plot(
            [anchor[0], anchor[0]],
            [anchor[1] + bar_space, anchor[1] + y_length_ax + bar_space],
            'k-',
            linewidth=linewidth,
            clip_on=False,
            solid_capstyle='butt',
            transform=ax.transAxes
        )

    # Do x scalebar.
    if not omit_x:

        if x_size is None:
            x_span = ax.get_xticks()[:2]
            x_length = x_span[1] - x_span[0]
            x_span_ax = ax.transLimits.transform(
                np.array([x_span, [0, 0]]).T)[:, 0]
        else:
            x_length = x_size
            x_span_ax = ax.transLimits.transform(
                np.array([[0, 0], [x_size, 0]]))[:, 0]
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
            Xx_bar_coords = [
                anchor[0] - x_length_ax - bar_space,
                anchor[0] - bar_space]
        else:
            Xx_text_coord = anchor[0] + x_length_ax / 2 + bar_space
            Xx_bar_coords = [
                anchor[0] + x_length_ax + bar_space,
                anchor[0] + bar_space]

        if usetex:
            x_label_text = '${}${}'.format(x_length, x_units)
        else:
            x_label_text = '{}{}'.format(x_length, x_units)

        ax.text(
            Xx_text_coord, anchor[1] + x_label_space,
            x_label_text,
            verticalalignment=verticalalignment, horizontalalignment='center',
            size='small', transform=ax.transAxes
        )

        # x scalebar
        ax.plot(
            Xx_bar_coords,
            [anchor[1], anchor[1]],
            'k-',
            linewidth=linewidth,
            clip_on=False,
            solid_capstyle='butt',
            transform=ax.transAxes
        )

    if remove_frame:
        ax.axis('off')


def hide_border(sides='a', trim=False, ax=None):
    """Remove and/or trim axes borders.

    Most common use is to remove top and right border from matplotlib axes.

    Arguments
    ---------
    sides : str `a`, one or more of `rltb`, or `none`
        Borders to remove. Use `a` for all, r` for right border, `rl` for right
        and left, etc.
    trim : bool, default False
        Shorten remaining axes to first and last tick. See `seborn.despine`.
    ax : matplotlib.axes object, defaults to current axes
        Hide border of this axes object.

    """
    # Check for correct input
    if not any([letter in sides for letter in 'arltb']):
        raise ValueError(
            'sides should be passed a string with `a` for all sides, '
            'or r/l/t/b as-needed for other sides.')

    if ax is None:
        ax = plt.gca()

    if sides == 'a':
        sides = 'rltb'

    # Remove ticks if needed.
    if 'l' in sides:
        ax.set_yticks([])
    if 'b' in sides:
        ax.set_xticks([])

    # Remove border(s); wraps seaborn.despine().
    sidekeys = {
        'r': 'right',
        'l': 'left',
        't': 'top',
        'b': 'bottom'
    }
    snsdespine_side_args = {}
    for key in sidekeys:
        if key in sides:
            snsdespine_side_args[sidekeys[key]] = True
        else:
            snsdespine_side_args[sidekeys[key]] = False
    sns.despine(trim=trim, ax=ax, **snsdespine_side_args)


def hide_ticks(ax=None):
    """Remove x and y ticks from axes object.

    Arguments
    ---------
    ax: matplotlib.axes, defaults to current axes

    """
    if ax is None:
        ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])


def p_to_string(p):
    """Convert p-value to pretty latex string.

    p is presented to three decimal places if p >= 0.05, and as
    p < 0.05/0.01/0.001 otherwise.

    Arguments
    ---------
    p: float
        p-value to format.

    Returns
    -------
        p-value formatted as a string.

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
