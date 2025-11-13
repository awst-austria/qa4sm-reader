# =========================
# texthelpers.py
# =========================

import qa4sm_reader.globals as globals
import numpy as np
import textwrap
import matplotlib.pyplot as plt
import warnings
import re


# =========================
# HELPER FUNCTIONS
# =========================

def wrapped_text(fig, text, width, fontsize) -> str:
    """
    Wrap a long string of text to fit into a given figure width.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object in which the text will be drawn.
    text : str
        The text to wrap.
    width : float
        The available width in pixels for the text.
    fontsize : int
        The font size in points used for estimating text width.

    Returns
    -------
    wrapped : str
        The text wrapped into multiple lines, separated by '\n'.
    """
    sample = "This is a very long text that should automatically wrap into multiple lines depending on the figure width"
    example_text = fig.suptitle(sample, fontsize=fontsize)

    renderer = fig.canvas.get_renderer()
    char_width_px = example_text.get_window_extent(renderer=renderer).width / len(sample)
    example_text.set_text("")

    # wrap text
    max_chars = int(width / char_width_px * 1)  # 1 ... factor, 0.8 would mean 80% of axwidth is maxlength of title
    wrapped = "\n".join(textwrap.wrap(text, max_chars))

    return wrapped

def best_legend_pos_exclude_list(ax, forbidden_locs= globals.leg_loc_forbidden):
    """
    Find the best legend position, excluding a list of positions.
    
    Parameters:
        ax : matplotlib.axes.Axes
        forbidden_locs : list of str or numbers, e.g. ["lower right", 2]
    
    Returns:
        best_loc_str : string of the best location
    """
    # standard Matplotlib positions
    locs = globals.leg_loc_dict
    
    # resolve forbidden positions to numbers
    forbidden_nums = set()
    for loc in forbidden_locs:
        if isinstance(loc, str):
            num = locs.get(loc)
            if num is not None:
                forbidden_nums.add(num)
        else:
            forbidden_nums.add(loc)
    
    # candidate positions
    candidate_locs = [loc for loc in locs.values() if loc not in forbidden_nums]
    
    fig = ax.figure
    
    min_overlap = float("inf")
    best_loc = candidate_locs[0]
    
    # evaluate overlap for each candidate
    leg = ax.get_legend()
    if not leg:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=re.escape("No artists with labels found to put in legend"),
                category=UserWarning,
            )
            leg = ax.legend()
    for loc in candidate_locs:
        leg.set_loc(loc=loc)
        fig.canvas.draw()
        
        bbox_legend = leg.get_window_extent()
        xdata = [line.get_xdata() for line in ax.get_lines()]
        ydata = [line.get_ydata() for line in ax.get_lines()]
        
        overlap = 0
        for xd, yd in zip(xdata, ydata):
            for x, y in zip(xd, yd):
                xpix, ypix = ax.transData.transform((x, y))
                if bbox_legend.contains(xpix, ypix):
                    overlap += 1
        
        if overlap < min_overlap:
            min_overlap = overlap
            best_loc = loc
    
    # convert numeric back to string
    best_loc_str = {v:k for k,v in locs.items()}[best_loc]
    return best_loc_str

def get_dataset_dict(Var) -> dict:
    """
    Creates dict containing dataset ids as keys and pretty name (version) as values.
    version only gets appended if there are multiple datasets witth the same pretty name.
    
    Parameters
    ----------
    Var : QA4SMMetricVariable
        Var in the image to make the map for.

    Returns
    -------
    d : dict
        Dict containing id:f"{pretty_name+(version)}" key-value pairs.
    """
    dataset_ref = {Var.Datasets.ref_id:f"{Var.Datasets.ref["pretty_name"]}"}
    dataset_others = {Var.Datasets.others_id[i]:f"{Var.Datasets.others[i]["pretty_name"]}" for i in range(len(Var.Datasets.others))}
    d = dataset_ref | dataset_others
    # Append version number if there are multiple datasets with the same pretty name
    groups = {}
    for k, v in (d).items():
        groups.setdefault(v, []).append(k)

    for i in groups.keys():
        if len(groups[i])>1: 
            for j in groups[i]:
                d[j] = d[j]+f" ({Var.Datasets.dataset_metadata(j)[1]["pretty_version"]})"
    return d

def get_legend_title(Var) -> str:
    """
    Creates a title for an existing legend from a QA4SMMetricVariable.

    Parameters
    ----------
    Var : QA4SMMetricVariable
        Var in the image to make the map for.

    Returns
    -------
    legend_title : String
        String containing the legend title
    """
    _, _, _, scale_ds, _ = Var.get_varmeta()
    d = get_dataset_dict(Var)
    
    # Append Unit
    for k in d.keys():
        d[k] = d[k]+f" [{Var.Datasets.dataset_metadata(k)[1]['mu'] if not scale_ds else scale_ds[1]["mu"]}]"
    
    legend_title = "Datasets:\n" + "\n".join(f"{k}: {v}" for k, v in (d).items())
    return legend_title

def append_legend_title(fig, ax, Var) -> tuple:
    """
    Appends a title to an existing legend from a QA4SMMetricVariable.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure that contains the axes and legend.
    ax : matplotlib.axes.Axes
        The subplot axis containing the legend to modify.
    Var : QA4SMMetricVariable
        Variable object used to construct the legend title.

    Returns
    -------
    fig, ax : tuple
        The same figure and axis, with the legend title updated.
    """  
    legend = ax.get_legend()
    legend_title = get_legend_title(Var)

    # Change Size to same as rest of legend
    if len(legend.legend_handles) == 0:
        legend.set_title(legend_title, prop={'size': globals.fontsize_legend})
    else:
        fs = legend.get_texts()[0].get_fontsize()
        legend.set_title(legend_title, prop={'size': fs})

    legend._legend_box.align = "left"
    best_loc_with_title = best_legend_pos_exclude_list(ax)

    # Step 4: move legend to new best position
    legend.set_loc(best_loc_with_title)

    return fig, ax

def smart_suptitle(fig, pad=globals.fontsize_title/2):
    """
    Compute position of Suptitle centeredd above axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object.
    pad : float
        Extra space (in fontsize) above the top axes title.
    """
    fig.canvas.draw() 

    top_positions = []
    for ax in fig.axes:
        if ax.get_visible():
            # get bounding box of the title in figure coordinates
            title = ax.title
            bbox = title.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_fig = bbox.transformed(fig.transFigure.inverted())
            top_positions.append(bbox_fig.y1 + title.get_fontsize()/(72*fig.get_figheight()))

    if top_positions:
        y = max(top_positions) + pad/(72*fig.get_figheight())
        y = min(y, 0.99) # So Suptitle always in Figure
    else:
        y = 0.99  # fallback if no axes
    # get center of axes or average of centers if multiple axes
    x = np.mean([(ax.get_position().x0+ax.get_position().x1)/2 for ax in fig.get_axes()[:globals.n_col_agg]])

    return x, y

def smart_suplabel(fig, axis, pad=globals.fontsize_label/2):
    """
    Compute position of suplabels centered according to axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object.
    axis : str
        Axis for which to calculate position.
    pad : float
        Extra space (in fontsize) above the top axes title.
    """
    fig.canvas.draw() 
    if axis == "x":
        bottom_positions = []
        for ax in fig.axes:
            if ax.get_visible():
                renderer = fig.canvas.get_renderer()

                # consider x-axis label and tick labels
                xlabel_bbox = ax.xaxis.label.get_window_extent(renderer=renderer)
                xtick_bboxes = [t.get_window_extent(renderer=renderer) for t in ax.xaxis.get_ticklabels() if t.get_text()]
                
                all_bboxes = [xlabel_bbox] + xtick_bboxes
                all_bboxes_fig = [b.transformed(fig.transFigure.inverted()) for b in all_bboxes]
                bottom_positions.append(min(b.y0 for b in all_bboxes_fig))

        if bottom_positions:
            y = min(bottom_positions) - globals.fontsize_label/(72*fig.get_figheight()) - pad/(72*fig.get_figheight())
        else:
            y = 0.01  # fallback
        x = np.mean([(ax.get_position().x0+ax.get_position().x1)/2 for ax in fig.get_axes()[:globals.n_col_agg]])
        return x, y
    elif axis == "y":
        left_positions = []
        for ax in fig.axes:
            if ax.get_visible():
                renderer = fig.canvas.get_renderer()

                # consider x-axis label and tick labels
                ylabel_bbox = ax.yaxis.label.get_window_extent(renderer=renderer)
                ytick_bboxes = [t.get_window_extent(renderer=renderer) for t in ax.yaxis.get_ticklabels() if t.get_text()]
                
                all_bboxes = [ylabel_bbox] + ytick_bboxes
                all_bboxes_fig = [b.transformed(fig.transFigure.inverted()) for b in all_bboxes]
                left_positions.append(min(b.x0 for b in all_bboxes_fig))

        if left_positions:
            x = min(left_positions) - globals.fontsize_label/(72*fig.get_figwidth()) - pad/(72*fig.get_figheight()) 
        else:
            x = 0.01  # fallback
        y = np.mean([(ax.get_position().y0+ax.get_position().y1)/2 for ax in fig.get_axes()[::globals.n_col_agg]])
        return x, y
    else: #fallback
        return 0.01, 0.01

def get_ax_width(fig) -> float:
    """Get horizontal distance of all axes in px. From left of first in row to right of last in row."""
    left = min([ax.get_position().x0 for ax in fig.get_axes()[:1]]) # Always the first ax
    right = max([ax.get_position().x1 for ax in fig.get_axes()])
    ax_width_px = fig.get_figwidth()*(right-left) * fig.dpi

    return ax_width_px

def get_ax_height(fig) -> float:
    """Get vertical distance of all axes in px. From bottom of column to top of column."""
    bottom = min([ax.get_position().y0 for ax in fig.get_axes()]) # Always the first ax
    top = max([ax.get_position().y1 for ax in fig.get_axes()])
    ax_height_px = fig.get_figheight()*(top-bottom) * fig.dpi

    return ax_height_px

def set_wrapped_title(fig, ax, title, fontsize=globals.fontsize_title,
                        pad=globals.title_pad, use_suptitle=False):
    """
    Set an axes or figure suptitle that automatically wraps to fit within figure width.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The target axes (ignored if use_suptitle=True).
    title : str
        The title text.
    fontsize : int
        Font size of the title.
    pad : float
        Extra spacing from the plot, in points.
    use_suptitle : bool, optional
        If True, set a figure-wide suptitle instead of an axes title.
    """
    fig.canvas.draw()  # needed to get renderer

    # width between most left point and most right point in axesin inches * dpi = pixels
    ax_width_px = get_ax_width(fig)

    # estimate character width (rough, depends on font)
    wrapped = wrapped_text(fig, title, ax_width_px, fontsize)

    if use_suptitle:
        x, y = smart_suptitle(fig)
        fig.suptitle(wrapped, fontsize=fontsize, y=y, x=x, ha="center", va="bottom")

    else:
        x, y = smart_suptitle(fig)
        ax.set_title(wrapped, fontsize=fontsize, pad=pad)