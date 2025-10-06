# =========================
# colors.py
# =========================

import hashlib
import qa4sm_reader.globals as globals

# =========================
# STATE VARIABLES
# =========================

# Shared dictionary storing assigned colors for each combo
color_map = {}

# Set of palette indices that are already used
used_indices = set()

# =========================
# PALETTE SETTINGS
# =========================

# Large reusable palette defined in globals
primary_palette = globals.color_palette_combinations
secondary_palette = globals.color_palette_combinations_2
palettes = [primary_palette, secondary_palette]
len_primary = len(primary_palette)
len_secondary = len(secondary_palette)
len_palettes = [len_primary, len_secondary]

# =========================
# HELPER FUNCTIONS
# =========================

def _deterministic_index(combo, max_colors=len_primary):
    """
    Compute a deterministic “order index” for a combo.

    Parameters
    ----------
    combo : hashable
        The dataset combination identifier (e.g., tuple, string, int)
    max_colors : int, optional
        Number of available colors in the palette, by default len_palette

    Returns
    -------
    int
        Deterministic index in the range [0, max_colors-1]

    Notes
    -----
    - Uses MD5 hash of the string representation of the combo.
    - Maps the hash to an integer index using modulo.
    """
    if max_colors <= 0:
        # No colors → fallback to 0 (caller must handle gracefully)
        return 0
    h = hashlib.md5(str(combo).encode()).hexdigest()
    return int(h, 16) % max_colors


def get_color_for(combo):
    """
    Return a consistent color for a given combo.

    Parameters
    ----------
    combo : hashable
        The dataset combination identifier

    Returns
    -------
    tuple
        RGB color tuple from palette, suitable for Seaborn or Matplotlib

    Logic
    -----
    1. If combo already has a color in `color_map`, return it.
    2. Otherwise, compute a preferred palette index via hashing.
    3. If that index is already used by another combo, move to the next free index
       (linear probing). Wrap around if necessary.
    4. Assign the selected color to the combo and mark the index as used.
    """
    if not any(len(pal) for pal in palettes):
        raise RuntimeError("No colors available: all palettes are empty.")
    
    if combo in color_map:
        return color_map[combo]

    # Determine preferred palette and index from hash
    idx_hash = _deterministic_index(combo)

    # Try each palette in order
    for palette_id, pal in enumerate(palettes):
        idx = idx_hash % len(pal)
        for _ in range(len(pal)):
            if (palette_id, idx) not in used_indices:
                used_indices.add((palette_id, idx))
                color_map[combo] = pal[idx]
                return pal[idx]
            idx = (idx + 1) % len(pal)

    # If all palettes exhausted, start a new cycle: clear used_indices and repeat
    used_indices.clear()
    palette_id = 0
    pal = palettes[palette_id]
    idx = idx_hash % len(pal)
    used_indices.add((palette_id, idx))
    color_map[combo] = pal[idx]
    return pal[idx]


def get_palette_for(combo):
    """
    Build a Seaborn-compatible palette dictionary for a sequence of combos.

    Parameters
    ----------
    combo : iterable
        Sequence of dataset combination identifiers

    Returns
    -------
    dict
        Mapping {combo: RGB tuple}, suitable for passing to Seaborn `palette` argument
    """
    if combo is None:
        return {}
    if len(combo) == 0:
        return {}
    return {c: get_color_for(c) for c in combo}
