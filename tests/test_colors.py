# test_colors.py
import pytest
import qa4sm_reader.colors as colors
import numpy as np


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test so tests are independent."""
    colors.color_map.clear()
    colors.used_indices.clear()
    yield
    colors.color_map.clear()
    colors.used_indices.clear()


def test_deterministic_index_is_stable():
    idx1 = colors._deterministic_index("comboA")
    idx2 = colors._deterministic_index("comboA")
    assert idx1 == idx2  # stable across calls


def test_deterministic_index_range():
    idx = colors._deterministic_index("comboB", max_colors=5)
    assert 0 <= idx < 5


def test_get_color_for_assigns_and_remembers():
    combo = "dataset1"
    color1 = colors.get_color_for(combo)
    assert combo in colors.color_map
    assert colors.color_map[combo] == color1

    # Should return the same color again
    color2 = colors.get_color_for(combo)
    assert color1 == color2


def test_get_color_for_collision_resolves():
    # Force two combos to map to the same initial index
    combo1 = "sameindex1"
    combo2 = "sameindex2"

    # Manually patch hash function to force collision
    original_func = colors._deterministic_index
    colors._deterministic_index = lambda combo, max_colors=len(colors.primary_palette): 0

    c1 = colors.get_color_for(combo1)
    c2 = colors.get_color_for(combo2)

    assert c1 != c2  # must probe to a different color

    colors._deterministic_index = original_func  # restore


def test_get_color_for_exhaustion_triggers_reset():
    # Fill all slots in the primary palette
    n = len(colors.primary_palette)
    combos = [f"combo{i}" for i in range(n)]
    for c in combos:
        colors.get_color_for(c)

    # Next combo will trigger exhaustion logic (but only for primary palette)
    new_combo = "combo_new"
    color = colors.get_color_for(new_combo)
    assert new_combo in colors.color_map
    assert len(colors.used_indices) > 0


def test_exhaustion_clears_and_reuses(monkeypatch):
    # Use tiny palettes to trigger exhaustion quickly
    fake_palette1 = ["red", "green"]
    fake_palette2 = ["blue", "yellow"]

    monkeypatch.setattr(colors, "primary_palette", fake_palette1)
    monkeypatch.setattr(colors, "secondary_palette", fake_palette2)
    monkeypatch.setattr(colors, "palettes", [fake_palette1, fake_palette2])

    # Reset state
    colors.color_map.clear()
    colors.used_indices.clear()

    # Fill both palettes completely
    combos = [f"c{i}" for i in range(len(fake_palette1) + len(fake_palette2))]
    for c in combos:
        colors.get_color_for(c)

    assert len(colors.used_indices) == 4  # all slots taken

    # Next request should trigger exhaustion path (clear + reuse)
    new_combo = "overflow"
    col = colors.get_color_for(new_combo)

    assert new_combo in colors.color_map
    assert len(colors.used_indices) == 1  # reset happened → only one index is used now
    assert col in fake_palette1  # resets always start from palette 0


def test_get_palette_for_multiple_combos():
    combos = ["c1", "c2", "c3"]
    pal = colors.get_palette_for(combos)
    assert isinstance(pal, dict)
    assert set(pal.keys()) == set(combos)
    for c in combos:
        assert pal[c] == colors.color_map[c]  # consistency


def test_deterministic_index_with_zero_colors():
    # Should not crash, but return 0
    assert colors._deterministic_index("anything", max_colors=0) == 0


def test_get_color_for_raises_if_all_palettes_empty(monkeypatch):
    monkeypatch.setattr(colors, "palettes", [[], []])
    monkeypatch.setattr(colors, "primary_palette", [])
    monkeypatch.setattr(colors, "secondary_palette", [])

    with pytest.raises(RuntimeError, match="No colors available"):
        colors.get_color_for("comboX")


def test_get_palette_for_empty_input():
    result = colors.get_palette_for([])
    assert result == {}

def test_get_palette_for_empty_numpy_array():
    result = colors.get_palette_for(np.array([]))
    assert result == {}
