"""Unit normalization and conversion helpers shared by the microscope managers."""
from __future__ import annotations

from collections.abc import Sequence

#: Spelling variants mapped to the unit names expected by NGFF.
_UNIT_ALIASES: dict[str, str] = {
    "um": "micrometer",
    "µm": "micrometer",  # micro sign
    "μm": "micrometer",  # greek mu
    "micron": "micrometer",
    "microns": "micrometer",
    "micrometers": "micrometer",
    "m": "meter",
    "meters": "meter",
    "mm": "millimeter",
    "millimeters": "millimeter",
    "nm": "nanometer",
    "nanometers": "nanometer",
    "s": "second",
    "sec": "second",
    "seconds": "second",
    "ms": "millisecond",
    "milliseconds": "millisecond",
    "min": "minute",
    "minutes": "minute",
    "h": "hour",
    "hours": "hour",
}

#: Multiplicative factors from a (normalized) length unit to micrometers.
_TO_MICROMETER: dict[str, float] = {
    "meter": 1e6,
    "millimeter": 1e3,
    "micrometer": 1.0,
    "nanometer": 1e-3,
}


def normalize_unit(unit: str | None) -> str | None:
    """Map common unit spellings (``um``, ``µm``, ``s`` ...) to NGFF unit names.

    Unknown units are returned unchanged (stripped); empty values give ``None``.
    """
    if not unit:
        return None
    unit = str(unit).strip()
    return _UNIT_ALIASES.get(unit, _UNIT_ALIASES.get(unit.lower(), unit))


def to_micrometers(
    scales: Sequence[float], units: Sequence[str | None]
) -> tuple[tuple[float, ...], tuple[str | None, ...]]:
    """Convert per-axis lengths to micrometers when their unit is known.

    Parameters
    ----------
    scales : sequence of float
        One length per spatial axis.
    units : sequence of str
        The unit of each entry in ``scales``.

    Returns
    -------
    (scales, units)
        Converted tuples. Entries with an unknown unit are left untouched.
    """
    if len(scales) != len(units):
        raise ValueError("'scales' and 'units' must have the same length.")

    out_scales, out_units = [], []
    for scale, unit in zip(scales, units):
        name = normalize_unit(unit)
        factor = _TO_MICROMETER.get(name)
        if factor is None:
            out_scales.append(scale)
            out_units.append(unit)
        else:
            out_scales.append(scale * factor)
            out_units.append("micrometer")
    return tuple(out_scales), tuple(out_units)
