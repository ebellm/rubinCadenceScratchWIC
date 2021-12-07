import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

# Turn the matplotlib version into a float
MATPLOTLIB_VERSION_FLOAT = float(
    "%s%s" % (matplotlib.__version__[0:3], matplotlib.__version__[3::].replace(".", ""))
)


def getColorbarLimits(cbar=None):

    """Gets the colorbar limits for matplotlib colorbar, respecting
    the matplotlib version"""

    if cbar is None:
        return 0.0, 1.0

    if MATPLOTLIB_VERSION_FLOAT < 3.3:
        return cbar.get_clim()
    else:
        return cbar.mappable.get_clim()


def showMollview(
    hparr: np.ndarray,
    nested=False,
    fignum=4,
    subplot=(1, 1, 1),
    figsize=(10, 6),
    cmap="Set2",
    numTicks=9,
    clobberFigure=True,
    sTitle: str = None,
    sUnit="TEST UNIT",
    sSuptitle: str = None,
    coord=["C", "G"],
    norm="linear",
    gratColor="0.2",
    gratAlpha=0.5,
    margins=(0.05, 0.05, 0.05, 0.05),
    fontsize: float = None,
    minval=None,
    maxval=None,
) -> matplotlib.figure.Figure:

    """Plot mollweide view of array of HEALPix data using customized colorbar ticks. Returns the figure.

    Args:
        hparr (np.ndarray): HEALPix array to show.
        nested (bool, optional): Whether the HEALPix scheme is NESTED, otherwise RING. Defaults ot False.
        fignum (int, optional): Matplotlib figure number. Defaults to 4.
        subplot (tuple, optional): Subplot to use. Defaults to (1,1,1).
        figsize (tuple, optional): Size of the figure in inches. Defaults to (10,6).
        cmap (str, optional): Colormap to use. Defaults to 'Set2'.
        numTicks (int, optional): Number of colorbar ticks. Defaults to 9.
        clobberFigure (bool, optional): Clear figure before plotting. Defaults to True.
        sTitle (str, optional): Title of the plot. Defaults to None.
        sUnit (str, optional): Unit of the data. Defaults to 'TEST UNIT'.
        sSuptitle (str, optional): Figure title. Defaults to None.
        coord (Union[str,List[str]], optional): Coordinate of the data (one of 'G','E','C') or
        transformation to use (list of two of the characters). Defaults to ['C','G'].
        norm (str, optional): Normalization of the colorbar, one of 'hist', 'log', 'linear. Defaults to
            'linear'.
        gratColor (str, optional): Graticule color. Defaults to '0.2'.
        gratAlpha (float, optional): Graticule alpha. Defaults to 0.5.
        margins (tuple, optional): Margins of the figure. Defaults to (0.05, 0.05, 0.05, 0.05).
        fontsize (float, optional): Size of the font. Defaults to None.
        minval (float, optional): Minimum value for the colorbar. Defaults to None.
        maxval (float, optional): Maximum value for the colorbar. Defaults to None.

    Returns:
        plt.Figure: Figure containing the mollweide view.
    """
    # the number of ticks and fontsize are overridden with
    # defaults if the colormap is one of the set below.
    Dnticks = {
        "Set1": 10,
        "Set2": 9,
        "Set3": 13,
        "tab10": 11,
        "Paired": 13,
        "Pastel2": 9,
        "Pastel1": 10,
        "Accent": 9,
        "Dark2": 10,
    }
    Dlsize = {
        "Set1": 9,
        "Set2": 10,
        "Set3": 7.5,
        "tab10": 9,
        "Paired": 7.5,
        "Pastel2": 9,
        "Pastel1": 9,
        "Accent": 9,
        "Dark2": 9,
    }

    # Set the number of ticks and the fontsize, allowing for
    # reversed colormaps
    labelsize = 8.0  # default
    cmapStem = cmap.split("_r")[0]
    if cmapStem in Dnticks.keys():
        numTicks = Dnticks[cmapStem]
        labelsize = Dlsize[cmapStem]

    # allow user-input override fontsize for labels
    if fontsize is not None:
        labelsize = fontsize

    # Is the input sensible?
    # if np.size(hparr) < 1:
    #     return None

    fig = plt.figure(fignum, figsize=figsize)
    if clobberFigure:
        fig.clf()

    hp.mollview(
        hparr,
        fignum,
        coord=coord,
        nest=nested,
        sub=subplot,
        title=sTitle,
        unit=sUnit,
        cmap=cmap,
        norm=norm,
        margins=margins,
        min=minval,
        max=maxval,
    )

    # Handle the colorbar
    cbar = plt.gca().images[-1].colorbar
    cmin, cmax = getColorbarLimits(cbar)
    # If it has log scale, cmin=0 is not valid.
    # This should be handled by mollview, if not cmin is replaced by the
    # smallest non-zero value of the array
    if cmin == 0 and norm == "log":
        cmin = np.amin(hparr[hparr != 0])
    # Set tick positions and labels
    cmap_ticks = np.linspace(cmin, cmax, num=numTicks)
    cbar.set_ticks(cmap_ticks, True)
    cmap_labels = ["{:5.0f}".format(t) for t in cmap_ticks]
    cbar.set_ticklabels(cmap_labels)
    cbar.ax.tick_params(labelsize=labelsize)
    # Change the position of the colorbar label
    text = [c for c in cbar.ax.get_children() if isinstance(c, matplotlib.text.Text) if c.get_text()][0]
    print(text.get_position())
    text.set_y(-3.0)  # valid for figsize=(8,6)

    # now show a graticule
    hp.graticule(color=gratColor, alpha=gratAlpha)

    # set supertitle if set
    if sSuptitle is not None:
        fig.suptitle(sSuptitle)

    return fig
