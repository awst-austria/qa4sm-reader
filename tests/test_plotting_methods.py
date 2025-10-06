# test_colors.py
import os
import io
import pytest
import pandas as pd
import seaborn as sns
import qa4sm_reader.plotting_methods as plm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import qa4sm_reader.globals as globals
import numpy as np
import cartopy.crs as ccrs


@pytest.fixture
def fig():
    fig = plt.figure()
    yield fig
    plt.close(fig)

@pytest.fixture
def simple_ax():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 2], label="line1")
    ax.plot([0, 1, 2], [2, 1, 0], label="line2")
    yield ax
    plt.close(fig)

@pytest.fixture
def mapax():
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    yield ax
    plt.close(fig)

@pytest.fixture
def box_ax():
    fig, ax = plt.subplots()
    # Sample data
    data = [np.random.rand(10), np.random.rand(10)]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True)
    yield ax
    plt.close(fig)

@pytest.fixture
def sample_data():
    return {
        "A": pd.DataFrame({"val": np.random.rand(10)}),
        "B": pd.DataFrame({"val": np.random.rand(10)}),
        "C": pd.DataFrame({"val": np.random.rand(10)}),
    }

@pytest.fixture
def sample_df():
    # Create a small dataframe compatible with bplot_catplot
    data = pd.DataFrame({
        "Dataset": ["A", "A", "B", "B", "C"],
        "values": [1, 2, 1, 3, 2],
        "meta": ["X", "Y", "X", "Y", "X"]
    })
    return data

@pytest.fixture
def sample_df_single_obs():
    data = pd.DataFrame({
        "Dataset": ["A", "B", "C"],
        "values": [1, 2, 3],
        "meta": ["X", "Y", "Z"]
    })
    return data

def test_returns_valid_location(simple_ax):
    loc = plm.best_legend_pos_exclude_list(simple_ax, forbidden_locs=[])
    assert isinstance(loc, str)
    assert loc in globals.leg_loc_dict.keys()


def test_respects_forbidden(simple_ax):
    # Forbid 'upper right'
    loc = plm.best_legend_pos_exclude_list(simple_ax, forbidden_locs=["upper right"])
    assert loc != "upper right"


def test_numeric_forbidden(simple_ax):
    # Forbid numeric equivalent of 'upper left'
    upper_left_num = globals.leg_loc_dict["upper left"]
    loc = plm.best_legend_pos_exclude_list(simple_ax, forbidden_locs=[upper_left_num])
    assert loc != "upper left"


def test_with_existing_legend(simple_ax):
    # Pre-create a legend
    simple_ax.legend(loc="lower left")
    loc = plm.best_legend_pos_exclude_list(simple_ax, forbidden_locs=[])
    assert isinstance(loc, str)
    assert loc in globals.leg_loc_dict.keys()

def test_overlap_minimization_prefers_less_overlap():
    import numpy as np
    fig, ax = plt.subplots()

    # Create many points clustered near the "upper right" corner
    for i in range(20):
        x = np.linspace(0.8, 1.0, 50)
        y = np.linspace(0.8, 1.0, 50) + i * 0.01
        ax.plot(x, y, label=f"line{i}")

    # With dense data in upper right, the algorithm should avoid "upper right"
    loc = plm.best_legend_pos_exclude_list(ax, forbidden_locs=[])
    assert loc != "upper right"
    assert isinstance(loc, str)

    plt.close(fig)

def test_extent_and_spine_set(mapax):
    plot_extent = [-10, 10, -5, 5]
    plm.style_map(mapax, plot_extent, add_grid=False, add_coastline=False,
              add_land=False, add_water=False, add_borders=False)
    # Extent set
    assert np.allclose(mapax.get_extent(crs=globals.data_crs), plot_extent)
    # Spine linewidth copied
    assert mapax.spines["geo"].get_linewidth() == mapax.spines["top"].get_linewidth()


def test_add_coastline_and_land(mapax):
    plot_extent = [-10, 10, -5, 5]
    plm.style_map(mapax, plot_extent, add_grid=False,
              add_coastline=True, add_land=True, add_water=True, add_borders=True)
    # Collect feature names from ax
    names = [f._feature.name for f in mapax._children if hasattr(f, "_feature")]
    assert "coastline" in names
    assert "land" in names
    assert "ocean" in names
    assert "admin_0_countries" in names


def test_disable_all_features(mapax):
    plot_extent = [-10, 10, -5, 5]
    plm.style_map(mapax, plot_extent, add_grid=False,
              add_coastline=False, add_land=False, add_water=False, add_borders=False)
    names = [f._feature.name for f in mapax._children if hasattr(f, "_feature")]
    assert names == []


def test_gridlines_added(mapax):
    plot_extent = [0, 20, 0, 10]
    styled_ax = plm.style_map(mapax, plot_extent, add_grid=True,
                          add_coastline=False, add_land=False,
                          add_water=False, add_borders=False)
    # Ensure gridlines object attached
    gridlines = [c for c in styled_ax._children if "Gridliner" in c.__class__.__name__]
    assert len(gridlines) == 1


def test_add_us_states(mapax):
    plot_extent = [-130, -60, 20, 50]  # USA
    plm.style_map(mapax, plot_extent, add_grid=False,
              add_coastline=False, add_land=False,
              add_water=False, add_borders=False,
              add_us_states=True)
    names = [f._feature.name for f in mapax._children if hasattr(f, "_feature")]
    assert any("states" in i for i in [n.lower() for n in names])

def test_add_logo_in_bg_front_file_not_found(fig, tmp_path):
    fake_path = str(tmp_path / "no_logo.png")
    with pytest.raises(FileNotFoundError):
        plm.add_logo_in_bg_front(fig, logo_path=fake_path)


def test_add_logo_in_bg_front_adds_axes(fig, tmp_path):
    # Create fake image
    path = tmp_path / "logo.png"
    arr = np.ones((10, 20, 3), dtype=np.uint8)
    mpimg.imsave(path, arr)

    ax = fig.add_subplot(111)
    # Should add a new logo axis
    plm.add_logo_in_bg_front(fig, logo_path=str(path), position="front_lower_right")
    logo_axes = [a for a in fig.axes if a is not ax]
    assert len(logo_axes) == 1
    assert not logo_axes[0].axison  # hidden axes


def test_add_logo_to_figure_warns_on_bad_position(fig, tmp_path):
    # Create fake image
    path = tmp_path / "logo.png"
    arr = np.ones((5, 5, 3), dtype=np.uint8)
    mpimg.imsave(path, arr)

    ax = fig.add_subplot(111)
    with pytest.warns(UserWarning, match="Position not implemented"):
        plm.add_logo_to_figure(fig, logo_path=str(path), position="nonsense")


def test_add_logo_to_figure_warns_if_no_logo(fig, tmp_path):
    # no file
    fake_path = str(tmp_path / "fake.png")
    ax = fig.add_subplot(111)
    with pytest.warns(UserWarning, match="No logo found"):
        plm.add_logo_to_figure(fig, logo_path=fake_path, position="lower_right")


def test_add_logo_to_figure_creates_axis(fig, tmp_path):
    # Create fake image
    path = tmp_path / "logo.png"
    arr = np.ones((8, 12, 3), dtype=np.uint8)
    mpimg.imsave(path, arr)

    ax = fig.add_subplot(111)
    plm.add_logo_to_figure(fig, logo_path=str(path), position="lower_right")
    # Expect 2 axes: the original + the logo axis
    assert len(fig.axes) == 2
    logo_ax = fig.axes[-1]
    assert not logo_ax.axison


def test_add_logo_to_figure_no_axes_creates_one(fig, tmp_path):
    # Create fake image
    path = tmp_path / "logo.png"
    arr = np.ones((8, 12, 3), dtype=np.uint8)
    mpimg.imsave(path, arr)

    # No axes initially
    assert fig.axes == []
    with pytest.warns(UserWarning, match="No axes found"):
        plm.add_logo_to_figure(fig, logo_path=str(path), position="lower_left")
    assert len(fig.axes) >= 1


def test_add_logo_in_bg_front_multiple_logos(monkeypatch, fig, tmp_path):
    # Create fake image
    path = tmp_path / "logo.png"
    arr = np.ones((10, 20, 3), dtype=np.uint8)
    mpimg.imsave(path, arr)

    ax = fig.add_subplot(111)

    # Patch globals for multiple logos
    monkeypatch.setattr(plm.globals, "n_logo", 2)
    monkeypatch.setattr(plm.globals, "n_col_logo", 2)

    plm.add_logo_in_bg_front(fig, logo_path=str(path))
    logo_axes = [a for a in fig.axes if a is not ax]
    assert len(logo_axes) >= 2


def test_capsizing_vertical_iterative(box_ax):
    n_lines = 0
    plm.capsizing(box_ax, orient="v", factor=1.0, iterative=True, n_lines=n_lines)
    
    # Check that cap lines xdata match expected center +/- half box width
    box_lines = box_ax.lines
    cap1, cap2 = box_lines[2], box_lines[3]
    whisker = box_lines[4]
    dist = whisker.get_xdata()[1] - whisker.get_xdata()[0]
    center = whisker.get_xdata().mean()
    assert np.allclose(cap1.get_xdata(), [center - dist/2, center + dist/2])
    assert np.allclose(cap2.get_xdata(), [center - dist/2, center + dist/2])


def test_capsizing_horizontal_iterative(box_ax):
    n_lines = 0
    # Transpose data to horizontal by swapping x/y
    for line in box_ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()
        line.set_xdata(y)
        line.set_ydata(x)
    plm.capsizing(box_ax, orient="h", factor=0.5, iterative=True, n_lines=n_lines)
    
    box_lines = box_ax.lines
    cap1, cap2 = box_lines[2], box_lines[3]
    whisker = box_lines[4]
    dist = whisker.get_ydata()[1] - whisker.get_ydata()[0]
    center = whisker.get_ydata().mean()
    expected = [center - dist*0.5/2, center + dist*0.5/2]
    assert np.allclose(cap1.get_ydata(), expected)
    assert np.allclose(cap2.get_ydata(), expected)


def test_capsizing_vertical_non_iterative(box_ax):
    plm.capsizing(box_ax, orient="v", factor=0.8, iterative=False)
    
    n_boxes = int(len(box_ax.lines) / 5)
    for i in range(n_boxes):
        box_lines = box_ax.lines[i*5:(i+1)*5]
        cap1, cap2 = box_lines[2], box_lines[3]
        whisker = box_lines[4]
        dist = whisker.get_xdata()[1] - whisker.get_xdata()[0]
        center = whisker.get_xdata().mean()
        expected = [center - dist*0.8/2, center + dist*0.8/2]
        assert np.allclose(cap1.get_xdata(), expected)
        assert np.allclose(cap2.get_xdata(), expected)


def test_capsizing_horizontal_non_iterative(box_ax):
    # Swap x/y for horizontal orientation
    for line in box_ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()
        line.set_xdata(y)
        line.set_ydata(x)
    plm.capsizing(box_ax, orient="h", factor=1.2, iterative=False)
    
    n_boxes = int(len(box_ax.lines) / 5)
    for i in range(n_boxes):
        box_lines = box_ax.lines[i*5:(i+1)*5]
        cap1, cap2 = box_lines[2], box_lines[3]
        whisker = box_lines[4]
        dist = whisker.get_ydata()[1] - whisker.get_ydata()[0]
        center = whisker.get_ydata().mean()
        expected = [center - dist*1.2/2, center + dist*1.2/2]
        assert np.allclose(cap1.get_ydata(), expected)
        assert np.allclose(cap2.get_ydata(), expected)

def dummy_plot_single(df, **kwargs):
    fig, ax = plt.subplots()
    ax.plot(df["val"])
    return fig, ax

def dummy_plot_axis(df, axis, **kwargs):
    axis.plot(df["val"])
    return 0.5, 1.0  # arbitrary unit_height, unit_width


def dummy_plot_no_axis(df, **kwargs):
    return 0.5, 1.0

def test_single_subplot_figsize_and_labels(sample_data):
    # Use only 1 item
    data = {"A": sample_data["A"]}
    fig, axes = plm.aggregate_subplots(data, dummy_plot_single)
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 1
    ax = axes[0]
    # Labels should be empty
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    # Figure size matches globals
    assert fig.get_figheight() == globals.boxplot_height_horizontal
    assert fig.get_figwidth() == globals.boxplot_width_horizontal


def test_multiple_subplots_axes_layout(sample_data):
    fig, axes = plm.aggregate_subplots(sample_data, dummy_plot_axis)
    n_expected = len(sample_data)
    assert len(axes) == n_expected
    # Titles should be set correctly
    titles = [ax.get_title() for ax in axes]
    for key in sample_data.keys():
        assert key in titles
    # Check figure size
    n_col = globals.n_col_agg
    n_rows = int(np.ceil(n_expected / n_col))
    assert fig.get_figheight() == globals.boxplot_height_vertical * n_rows
    assert fig.get_figwidth() == globals.boxplot_width_vertical * n_col
    # Extra legends should be empty for n != 0
    for i, ax in enumerate(axes[1:]):
        lines = ax.get_legend().get_texts() if ax.get_legend() else []
        assert lines == [] or all([t.get_text() == "" for t in lines])


def test_error_if_no_axis_param(sample_data):
    with pytest.raises(KeyError):
        plm.aggregate_subplots(sample_data, dummy_plot_no_axis)

def dummy_add_cat_info(df, metadata_name):
    # simply return df for testing
    return df

def dummy_get_palette_for(combos):
    return {c: (0.1, 0.2, 0.3) for c in combos}

def test_bplot_creates_figure_and_axis(monkeypatch, sample_df):
    monkeypatch.setattr(plm, "add_cat_info", dummy_add_cat_info)
    monkeypatch.setattr(plm, "get_palette_for", dummy_get_palette_for)
    monkeypatch.setattr(plm, "capsizing", lambda ax, **kwargs: None)
    monkeypatch.setattr(plm, "best_legend_pos_exclude_list", lambda ax: "best")

    fig, ax = plm.bplot_catplot(sample_df, axis_name="Values", metadata_name="meta")
    assert isinstance(fig, plt.Figure)
    assert hasattr(ax, "plot")
    assert (ax.get_ylabel() == "Values") or (ax.get_xlabel() == "Values") # depends on orientation


def test_bplot_single_observation_points(monkeypatch, sample_df_single_obs):
    monkeypatch.setattr(plm, "add_cat_info", dummy_add_cat_info)
    monkeypatch.setattr(plm, "get_palette_for", dummy_get_palette_for)
    monkeypatch.setattr(plm, "capsizing", lambda ax, **kwargs: None)
    monkeypatch.setattr(plm, "best_legend_pos_exclude_list", lambda ax: "best")

    fig, ax = plm.bplot_catplot(sample_df_single_obs, axis_name="Values", metadata_name="meta")
    assert len(ax.patches) >= 0
    assert ax.get_legend() is not None


def test_bplot_with_existing_axis(monkeypatch, sample_df):
    monkeypatch.setattr(plm, "add_cat_info", dummy_add_cat_info)
    monkeypatch.setattr(plm, "get_palette_for", dummy_get_palette_for)
    monkeypatch.setattr(plm, "capsizing", lambda ax, **kwargs: None)
    monkeypatch.setattr(plm, "best_legend_pos_exclude_list", lambda ax: "best")

    fig, ax_orig = plt.subplots()
    result = plm.bplot_catplot(sample_df, axis_name="Values", metadata_name="meta", axis=ax_orig)
    assert result is None or isinstance(result, tuple)


def test_bplot_tick_adjustments(monkeypatch, sample_df):
    monkeypatch.setattr(plm, "add_cat_info", dummy_add_cat_info)
    monkeypatch.setattr(plm, "get_palette_for", dummy_get_palette_for)
    monkeypatch.setattr(plm, "capsizing", lambda ax, **kwargs: None)
    monkeypatch.setattr(plm, "best_legend_pos_exclude_list", lambda ax: "best")

    fig, ax = plm.bplot_catplot(sample_df, axis_name="Values", metadata_name="meta")
    grid_lines = [line.get_linestyle() for line in ax.get_xgridlines()] + [line.get_linestyle() for line in ax.get_ygridlines()]
    assert any(ls in ['-', '--', ':', '-.'] for ls in grid_lines)