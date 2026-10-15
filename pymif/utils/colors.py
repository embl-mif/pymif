"""Channel color parsing shared by the readers, the Zarr writer and napari."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

_HEX_COLOR_PATTERN = re.compile(r"^#?[0-9a-fA-F]{6}$")
_ARGB_PATTERN = re.compile(r"^[0-9a-fA-F]{8}$")

#: Default colors assigned to channels that do not declare one (cycled).
DEFAULT_CHANNEL_PALETTE = ("FFFFFF", "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF")


def default_channel_color(index: int) -> str:
    """Return the default ``RRGGBB`` color for channel ``index`` (cycles the palette)."""
    return DEFAULT_CHANNEL_PALETTE[index % len(DEFAULT_CHANNEL_PALETTE)]


def parse_color(value: str) -> str:
    """Parse a color to a six-digit uppercase hex string (without ``#``).

    Accepts 6-digit hex codes (``#`` optional) and matplotlib color names
    (e.g. ``"magenta"``, ``"cyan"``).
    """
    from matplotlib.colors import cnames

    if not isinstance(value, str):
        raise TypeError("Channel colors must be strings.")

    value = value.strip()
    if _HEX_COLOR_PATTERN.match(value):
        return value.replace("#", "").upper()

    lower = value.lower()
    if lower in cnames:
        return cnames[lower].replace("#", "").upper()

    raise TypeError(
        f"Invalid color {value!r}. Use a 6-digit hex code or a valid "
        "matplotlib color name."
    )


def ome_color_to_hex(value: int | str) -> str:
    """Convert an OME-XML ``Color`` (signed 32-bit RGBA integer) to ``RRGGBB``."""
    return f"{(int(value) & 0xFFFFFFFF) >> 8:06X}"


def parse_channel_color(raw: Any, index: int = 0) -> str:
    """Convert a vendor-specific channel color to ``RRGGBB``.

    This is the single entry point used by the microscope managers, the Zarr
    writer and the napari viewer. It accepts:

    * ``None`` / empty values -> the default palette color for ``index``;
    * integers or integer strings: 24-bit RGB (``0 <= v <= 0xFFFFFF``, as written
      by Opera) or signed 32-bit OME ``RGBA`` (see :func:`ome_color_to_hex`);
    * ``0x``-prefixed hex strings;
    * ``#AARRGGBB`` / ``AARRGGBB`` (Zeiss) -> alpha dropped;
    * 6-digit hex codes and matplotlib color names (see :func:`parse_color`).
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default_channel_color(index)

    if isinstance(raw, (int, np.integer)) and not isinstance(raw, bool):
        value = int(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        try:
            value = int(text, 16) if text.lower().startswith("0x") else int(text)
        except ValueError:
            digits = text.lstrip("#")
            if _ARGB_PATTERN.match(digits):
                return digits[2:].upper()
            return parse_color(text)
    else:
        raise TypeError(f"Unsupported channel color {raw!r}.")

    if 0 <= value <= 0xFFFFFF:
        return f"{value:06X}"
    return ome_color_to_hex(value)


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert ``RRGGBB`` to an ``(r, g, b)`` tuple of floats in ``[0, 1]``."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
