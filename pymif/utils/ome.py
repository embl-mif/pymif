"""Minimal OME-XML helpers shared by the OME-TIFF based managers."""
from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from .metadata import ChannelInfo
from .ngff import ome_color_to_hex

T = TypeVar("T")

#: Namespace-agnostic prefix, so any OME schema version is accepted.
_NS = "{*}"


def xml_attr(node: ET.Element, name: str, cast: Callable[[str], T] = str, *,
             default: Any = ..., what: str = "") -> T:
    """Read and convert an XML attribute.

    Raises a ``ValueError`` naming the missing attribute when no ``default`` is
    given, instead of a bare ``KeyError``.
    """
    if name in node.attrib:
        return cast(node.attrib[name])
    if default is ...:
        raise ValueError(f"Missing required attribute '{name}'{(' in ' + what) if what else ''}.")
    return default


@dataclass
class OmePixels:
    """Subset of the OME ``<Pixels>`` element used by PyMIF."""

    size_t: int
    size_c: int
    size_z: int
    size_y: int
    size_x: int
    scale_zyx: tuple[float, float, float]
    units_zyx: tuple[str, str, str]
    time_increment: float
    time_increment_unit: str
    dtype: str
    channels: list[ChannelInfo] = field(default_factory=list)
    tiffdata: list[ET.Element] = field(default_factory=list)


def _ome_channel_color(raw: str | None, color_format: str) -> str | None:
    """Convert a ``<Channel Color=...>`` attribute according to ``color_format``."""
    if color_format == "ome":
        # Spec-compliant: signed 32-bit RGBA integer, -1 (opaque white) if absent.
        try:
            return ome_color_to_hex(raw if raw is not None else -1)
        except ValueError:
            return raw  # not an integer: a name or hex code
    return raw  # "auto": resolved later by parse_channel_color


def parse_ome_xml(xml: str | bytes | ET.Element, color_format: str = "ome") -> OmePixels:
    """Parse the first ``<Pixels>`` element of an OME-XML document.

    Missing physical pixel sizes fall back to ``1.0`` with a warning so that
    the user knows the voxel size was not recorded in the file.

    Parameters
    ----------
    xml : str, bytes or Element
        OME-XML document (or its root element).
    color_format : {"ome", "auto"}
        ``"ome"`` (default) reads integer channel colors as signed 32-bit RGBA
        as the OME schema prescribes. ``"auto"`` additionally accepts plain
        24-bit RGB integers, which some writers (Opera) use, and leaves absent
        colors to the default palette.
    """
    if color_format not in ("ome", "auto"):
        raise ValueError("color_format must be 'ome' or 'auto'.")
    root = ET.fromstring(xml) if isinstance(xml, (str, bytes)) else xml
    pixels = root if root.tag.endswith("Pixels") else root.find(f".//{_NS}Pixels")
    if pixels is None:
        raise ValueError("No <Pixels> element found in the OME-XML metadata.")

    scales, units = [], []
    fallback_unit = pixels.attrib.get("PhysicalSizeXUnit", "µm")
    for axis in "ZYX":
        if f"PhysicalSize{axis}" in pixels.attrib:
            scales.append(float(pixels.attrib[f"PhysicalSize{axis}"]))
        else:
            warnings.warn(f"OME-XML has no PhysicalSize{axis}; assuming 1.0.", stacklevel=3)
            scales.append(1.0)
        units.append(pixels.attrib.get(f"PhysicalSize{axis}Unit", fallback_unit))

    channels = [
        ChannelInfo(ch.attrib.get("Name") or f"Channel {i}", _ome_channel_color(ch.attrib.get("Color"), color_format))
        for i, ch in enumerate(pixels.findall(f"{_NS}Channel"))
    ]
    size_c = int(pixels.attrib.get("SizeC", max(1, len(channels))))
    while len(channels) < size_c:
        channels.append(ChannelInfo(f"Channel {len(channels)}"))

    return OmePixels(
        size_t=int(pixels.attrib.get("SizeT", 1)),
        size_c=size_c,
        size_z=int(pixels.attrib.get("SizeZ", 1)),
        size_y=xml_attr(pixels, "SizeY", int, what="<Pixels>"),
        size_x=xml_attr(pixels, "SizeX", int, what="<Pixels>"),
        scale_zyx=tuple(scales),
        units_zyx=tuple(units),
        time_increment=float(pixels.attrib.get("TimeIncrement", 1)),
        time_increment_unit=pixels.attrib.get("TimeIncrementUnit", "s"),
        dtype=pixels.attrib.get("Type", "uint16"),
        channels=channels,
        tiffdata=pixels.findall(f"{_NS}TiffData"),
    )
