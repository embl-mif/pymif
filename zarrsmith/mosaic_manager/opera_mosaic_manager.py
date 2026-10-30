from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import unescape

import numpy as np
from tqdm.auto import tqdm

from .fields import FIELD_PREFIX, connected_components
from .mosaic import MosaicManager
from .tiff_array import TiffStackArray
from .tile import Tile

logger = logging.getLogger("zarrsmith")

#: Names of the image index written by Harmony exports, in order of preference.
INDEX_NAMES = ("Index.xml", "Index.idx.xml")

#: Micrometres per unit of the ``Unit`` attributes in the index.
_UNITS = {"m": 1e6, "mm": 1e3, "um": 1.0, "µm": 1.0, "µm": 1.0, "nm": 1e-3}

# The index of a large plate is hundreds of MB: it is scanned in blocks with regular
# expressions (about 5x faster than an XML parser), keeping only what the tiles need.
_BLOCKS = re.compile(
    rb'<Entry\s+ChannelID="(\d+)"[^>]*>(.*?)</Entry>'  # channel settings (Maps)
    rb"|<Image(?:\s[^>]*[^/>])?>(.*?)</Image>",  # one plane of one channel (Images); skips <Image id=".."/>
    re.S,
)
_OPENING = re.compile(rb"<(?:Entry|Image)(?=[\s>/])")
_TAG = re.compile(rb"<(\w+)([^>]*)>([^<]*)</\1>")
_UNIT = re.compile(rb'Unit="([^"]*)"')
_PLATE = re.compile(rb"<PlateRows>(\d+)</PlateRows>.*?<PlateColumns>(\d+)</PlateColumns>", re.S)
_READ_BLOCK = 1 << 24

#: Well pitch (micrometres) of the standard (ANSI/SLAS) plate formats, by ``(rows, columns)``.
#: Harmony gives positions relative to each well centre, not the plate geometry.
WELL_PITCH = {(2, 3): 39120.0, (3, 4): 26010.0, (4, 6): 19300.0, (6, 8): 13080.0,
              (8, 12): 9000.0, (16, 24): 4500.0, (32, 48): 2250.0}


# ---------------------------------------------------------------------------
# flat-field
# ---------------------------------------------------------------------------

def parse_acapella(text: str) -> Any:
    """Parse the ``{Key: value, ...}`` notation of Harmony / Acapella metadata (``FlatfieldProfile``).

    Mappings ``{...}``, lists ``[...]`` and scalars (numbers, or text up to the next
    ``,``, ``}`` or ``]``, such as ``Acapella:2013``) are supported.
    """
    pos = 0

    def skip():
        nonlocal pos
        while pos < len(text) and text[pos].isspace():
            pos += 1

    def value():
        nonlocal pos
        skip()
        if text[pos] == "{":
            pos += 1
            out = {}
            skip()
            if text[pos] == "}":
                pos += 1
                return out
            while True:
                end = text.index(":", pos)
                key = text[pos:end].strip()
                pos = end + 1
                out[key] = value()
                skip()
                pos += 1
                if text[pos - 1] == "}":
                    return out
        if text[pos] == "[":
            pos += 1
            out = []
            skip()
            if text[pos] == "]":
                pos += 1
                return out
            while True:
                out.append(value())
                skip()
                pos += 1
                if text[pos - 1] == "]":
                    return out
        start = pos
        while pos < len(text) and text[pos] not in ",}]":
            pos += 1
        word = text[start:pos].strip()
        for cast in (int, float):
            try:
                return cast(word)
            except ValueError:
                pass
        return word

    try:
        return value()
    except (IndexError, ValueError) as e:
        raise ValueError(f"Cannot parse Harmony profile {text[:80]!r}...") from e


class OperaFlatfield:
    """Harmony's background flat-field correction of one channel (picklable).

    Harmony describes the uneven background of a channel as a 2D polynomial ``P``,
    normalised to a mean of 1, and its mean level ``Mean`` (``FlatfieldProfile`` of
    the index, ``Background`` entry). It corrects every plane with::

        raw >= Mean * P :  raw - Mean * (P - 1)    (above the background: remove its unevenness)
        raw <  Mean * P :  raw / P                 (below it: rescale it to the flat level)

    Both branches give ``Mean`` at ``raw = Mean * P``. This reproduces the images
    stitched by Harmony to within rounding.

    ``P(u, v) = sum_k sum_i c[k][i] * u**(k - i) * v**i``, with ``u = (x - Origin[0]) * Scale[0]``
    along image columns and ``v = (y - Origin[1]) * Scale[1]`` along rows, in the
    pixels of a ``Dims`` image (rescaled to the image size, e.g. for another binning).
    """

    def __init__(self, mean: float, coefficients, dims=None, origin=None, scale=None):
        self.mean = float(mean)
        self.coefficients = [[float(c) for c in row] for row in coefficients]
        self.dims = None if dims is None else tuple(float(d) for d in dims)
        self.origin = None if origin is None else tuple(float(o) for o in origin)
        self.scale = None if scale is None else tuple(float(s) for s in scale)
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}

    @classmethod
    def from_profile(cls, profile, channel: str = "") -> Optional["OperaFlatfield"]:
        """Correction described by a ``FlatfieldProfile`` (text or parsed), or ``None``
        when it leaves the images as they are (``Identity`` profile, ``Null`` character)."""
        if isinstance(profile, str):
            profile = parse_acapella(profile)
        if not isinstance(profile, dict):
            raise ValueError(f"Unexpected flat-field profile: {profile!r}")
        foreground = profile.get("Foreground")
        if isinstance(foreground, dict) and (foreground.get("Profile") or {}).get("Type") not in (None, "Identity"):
            logger.warning("Channel %s has a Foreground flat-field profile: only the Background "
                           "correction is applied.", channel)
        background = profile.get("Background")
        if not isinstance(background, dict) or background.get("Character") == "Null":
            return None
        shape = background.get("Profile") or {}
        kind = shape.get("Type")
        if kind in (None, "Identity"):
            return None
        if kind != "Polynomial":
            raise ValueError(f"Channel {channel}: flat-field profile type {kind!r} is not supported.")
        return cls(background["Mean"], shape["Coefficients"], shape.get("Dims"), shape.get("Origin"),
                   shape.get("Scale"))

    def profile(self, shape: Tuple[int, int]) -> np.ndarray:
        """``P`` for an image of ``shape`` (rows, columns), as float32."""
        shape = (int(shape[0]), int(shape[1]))
        if shape not in self._cache:
            ny, nx = shape
            dims = self.dims or (nx, ny)
            origin = self.origin or ((dims[0] - 1) / 2, (dims[1] - 1) / 2)
            scale = self.scale or (1 / dims[0], 1 / dims[1])
            # pixel centres of this image in the pixels of the profile image
            x = (np.arange(nx) + 0.5) * dims[0] / nx - 0.5
            y = (np.arange(ny) + 0.5) * dims[1] / ny - 0.5
            u = ((x - origin[0]) * scale[0])[None, :]
            v = ((y - origin[1]) * scale[1])[:, None]
            p = np.zeros(shape)
            for k, row in enumerate(self.coefficients):
                for i, c in enumerate(row):
                    p += c * u ** (k - i) * v ** i
            self._cache[shape] = np.maximum(p, 1e-3).astype(np.float32)  # guard the division
        return self._cache[shape]

    def __call__(self, plane: np.ndarray) -> np.ndarray:
        p = self.profile(plane.shape)
        raw = plane.astype(np.float32)
        background = self.mean * p
        return np.where(raw >= background, raw - (background - self.mean), raw / p)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"] = {}  # rebuilt where it is used
        return state

    def __repr__(self) -> str:
        return f"OperaFlatfield(mean={self.mean}, degree={len(self.coefficients) - 1}, dims={self.dims})"


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def _well_name(row: int, col: int) -> str:
    """``(1, 3)`` -> ``"A3"``, ``(27, 1)`` -> ``"AA1"``."""
    letters = ""
    while row > 0:
        row, rest = divmod(row - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return f"{letters}{col}"


def _tags(block: bytes) -> Dict[str, Tuple[str, Optional[str]]]:
    """``{tag: (text, unit)}`` of the simple child elements of an index block."""
    out = {}
    for name, attrs, text in _TAG.findall(block):
        unit = _UNIT.search(attrs)
        out[name.decode()] = (unescape(text.decode("utf-8")), unit.group(1).decode("utf-8") if unit else None)
    return out


def _micrometres(text, unit=None, default_unit: str = "m") -> float:
    """Value of a position or resolution in micrometres (``Unit`` attribute, metres by default)."""
    unit = (unit.decode("utf-8") if isinstance(unit, bytes) else unit) or default_unit
    if unit not in _UNITS:
        raise ValueError(f"Unknown unit {unit!r} in the Opera index.")
    return float(text) * _UNITS[unit]


def _position(tags, name: bytes) -> float:
    if name not in tags:
        return 0.0
    attrs, text = tags[name]
    unit = _UNIT.search(attrs)
    return _micrometres(text, unit.group(1) if unit else None)


def read_index(path: str | Path, progress: bool = True):
    """Channels and planes of a Harmony ``Index.xml``.

    Returns ``(channels, stacks)``:

    * ``channels`` maps a channel id to its settings (``ChannelName``,
      ``ImageResolutionX``, ``OrientationMatrix``, ``FlatfieldProfile``, ...), as
      ``{tag: (text, unit)}``;
    * ``stacks`` maps ``(row, col, field, timepoint, channel)`` to
      ``{"x": um, "y": um, "planes": [(plane id, z um, file name or None), ...]}``,
      with ``None`` for planes whose ``State`` is not ``Ok``.
    """
    path = Path(path)
    channels: Dict[int, Dict[str, Tuple[str, Optional[str]]]] = {}
    stacks: Dict[tuple, dict] = {}
    with open(path, "rb") as f, tqdm(total=path.stat().st_size, unit="B", unit_scale=True,
                                     desc=f"reading {path.name}", disable=not progress) as bar:
        buf = b""
        while True:
            chunk = f.read(_READ_BLOCK)
            bar.update(len(chunk))
            buf += chunk
            last = 0
            for m in _BLOCKS.finditer(buf):
                last = m.end()
                if m.group(1) is not None:
                    channels.setdefault(int(m.group(1)), {}).update(_tags(m.group(2)))
                    continue
                # tag names and values stay bytes: hundreds of thousands of planes in large plates
                t = {name: (attrs, text) for name, attrs, text in _TAG.findall(m.group(3))}
                if b"URL" not in t:
                    continue
                key = (int(t[b"Row"][1]), int(t[b"Col"][1]), int(t[b"FieldID"][1]),
                       int(t[b"TimepointID"][1]) if b"TimepointID" in t else 1, int(t[b"ChannelID"][1]))
                stack = stacks.get(key)
                if stack is None:
                    stack = stacks[key] = {"x": _position(t, b"PositionX"), "y": _position(t, b"PositionY"),
                                           "planes": []}
                ok = t[b"State"][1] == b"Ok" if b"State" in t else True
                stack["planes"].append((int(t[b"PlaneID"][1]), _position(t, b"PositionZ"),
                                        unescape(t[b"URL"][1].decode("utf-8")) if ok else None))
            if not chunk:
                break
            # keep the unfinished block at the end of the buffer: from its opening tag
            # (not <ImageSizeX> & co.), or the last bytes, which may hold a cut opening tag
            tail = buf[last:]
            starts = [o.start() for o in _OPENING.finditer(tail)]
            buf = tail[starts[-1]:] if starts else tail[-16:]
    return channels, stacks


def read_plate(path: str | Path) -> Optional[Tuple[int, int]]:
    """``(PlateRows, PlateColumns)`` of a Harmony index (in its first block), or ``None``."""
    with open(path, "rb") as f:
        m = _PLATE.search(f.read(_READ_BLOCK))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _orientation(settings) -> Optional[List[List[float]]]:
    text = settings.get("OrientationMatrix", (None, None))[0]
    return json.loads(text) if text else None


# ---------------------------------------------------------------------------
# mosaic
# ---------------------------------------------------------------------------

class OperaMosaicManager(MosaicManager):
    """Raw Opera Phenix / Operetta (Revvity Harmony) tiles, before any stitching.

    Not to be confused with :class:`~zarrsmith.microscope_manager.OperaManager`, which
    reads images stitched by Harmony (``stitchresult/*.ome.tiff``). This manager reads
    a Harmony export: ``Images/Index.xml`` and one TIFF file per plane
    (``r01c03f01p01-ch1sk1fk1fl1.tiff``: row 1, column 3, Opera field 1, plane 1, channel 1).
    Every Opera field of every channel is one tile; its planes stay on disk until read.

    * ``channel``: the ``ChannelName`` of the index;
    * ``field``: tiles are grouped per well into fields of touching tiles, named
      ``<well>_field<k>`` (``A3_field0``, ``A3_field1``, ...);
    * ``timepoint``: ``TimepointID - 1`` (mosaics with several timepoints are refused for now);
    * ``attrs``: ``stack`` (``r01c03f01``, the tile id without the channel), ``well``,
      ``row``, ``col``, ``opera_field``, ``channel_id``, ``flatfield`` and ``folder``.

    Geometry, consistent with the images stitched by Harmony: the x/y pixel size is the
    ``ImageResolution`` of the index scaled by the channel's ``OrientationMatrix``
    (a calibration of a few percent, which Harmony applies to its stitched images),
    tiles are centred on the field position, and stage ``y`` runs against image rows
    when the matrix says so (negative ``y`` scale). Planes are placed at their
    ``PositionZ``, without the matrix's ``z`` scale and offset.

    Stage positions are relative to the centre of each well, so every well is shifted
    to its place on the plate (A1 at the top left, ``y`` along rows, ``x`` along columns),
    by the well pitch of the plate format (``PlateRows`` x ``PlateColumns``, see
    :data:`WELL_PITCH`). The fused images do not depend on it.

    With ``flatfield=True`` (default), every plane is corrected with the background
    flat-field profile stored by Harmony (see :class:`OperaFlatfield`), as in the
    Harmony stitched images. The profiles are read from the index, or from
    ``FFC_Profile/*.xml`` next to ``Images/`` when the index has none.
    """

    def __init__(self, path, flatfield: bool = True, progress: bool = True):
        """
        Parameters
        ----------
        path
            Harmony export folder (containing ``Images/``), the ``Images`` folder, or the index file.
        flatfield
            Apply Harmony's flat-field correction to every plane.
        progress
            Show a progress bar while reading the index (hundreds of MB for large plates).
        """
        self.flatfield = bool(flatfield)
        self.progress = progress
        self.flatfields: Dict[str, Optional[OperaFlatfield]] = {}
        super().__init__(path)

    @staticmethod
    def find_index(path: str | Path) -> Optional[Path]:
        """The Harmony index of an export folder, its ``Images`` folder or the index itself."""
        path = Path(path)
        if path.is_file():
            return path
        for folder in (path, path / "Images"):
            for name in INDEX_NAMES:
                if (folder / name).is_file():
                    return folder / name
        return None

    def _profiles(self, index: Path, channels) -> Dict[int, str]:
        """``FlatfieldProfile`` text of every channel: from the index, else from ``FFC_Profile/``."""
        found = {c: s["FlatfieldProfile"][0] for c, s in channels.items() if "FlatfieldProfile" in s}
        if found:
            return found
        root = index.parent.parent if index.parent.name.lower() == "images" else index.parent
        for xml in sorted((root / "FFC_Profile").glob("*.xml")) if (root / "FFC_Profile").is_dir() else []:
            text = xml.read_bytes()
            for m in _BLOCKS.finditer(text):
                if m.group(1) is not None and b"FlatfieldProfile" in m.group(2):
                    found.setdefault(int(m.group(1)), _tags(m.group(2))["FlatfieldProfile"][0])
            if found:
                logger.info("Flat-field profiles read from %s", xml)
                return found
        return found

    def _scan(self) -> List[Tile]:
        index = self.find_index(self.path)
        if index is None:
            raise FileNotFoundError(f"No Harmony index ({' or '.join(INDEX_NAMES)}) in {self.path} or its Images folder.")
        channels, stacks = read_index(index, progress=self.progress)
        if not stacks:
            return []
        folder = index.parent

        names = self._channel_names(channels, {key[4] for key in stacks})
        voxel_yx = self._pixel_sizes(channels, names)
        flips = {c: self._flips(channels.get(c, {})) for c in names}

        profiles = self._profiles(index, channels) if self.flatfield else {}
        corrections = {}
        for c, name in names.items():
            corrections[c] = OperaFlatfield.from_profile(profiles[c], name) if c in profiles else None
            self.flatfields[name] = corrections[c]
            if self.flatfield and c not in profiles:
                logger.info("Channel %s has no flat-field profile: it is not corrected.", name)

        # planes of every (well, field, timepoint, channel), sorted along z; a repeated
        # plane id (e.g. several FLIM images) keeps its first image
        for s in stacks.values():
            s["planes"] = sorted({p[0]: p for p in reversed(s["planes"])}.values(), key=lambda p: p[1])
        steps = [np.diff([p[1] for p in s["planes"]]) for s in stacks.values()]
        steps = np.concatenate(steps) if steps else np.array([])
        dz = float(np.median(steps)) if steps.size else 1.0
        if dz <= 0:
            raise ValueError("Opera planes of one field share the same PositionZ: cannot place them along z.")
        if steps.size and np.max(np.abs(steps - dz)) > 1e-3 * dz:
            logger.warning("Opera planes are not evenly spaced (steps %s um): using %.4g um.",
                           sorted(set(np.round(steps, 4).tolist())), dz)
        shape_yx = {c: self._image_size(channels.get(c, {})) for c in names}
        sample = next((p[2] for s in stacks.values() for p in s["planes"] if p[2] is not None), None)
        if sample is None:
            raise FileNotFoundError(f"Every plane of {index} is marked as failed.")
        dtype, first_shape = self._header(folder, sample)
        py, px = voxel_yx
        sizes = [shape_yx[c] or first_shape for c in names]
        pitch = self._well_pitch(read_plate(index), stacks, max(max(ny * py, nx * px) for ny, nx in sizes))

        tiles = []
        missing = 0
        for (row, col, field, timepoint, c), s in sorted(stacks.items()):
            members = s["planes"]
            missing += sum(p[2] is None for p in members)
            ny, nx = shape_yx[c] or first_shape
            sy, sx = flips[c]
            # stage positions are relative to the well centre: place the well on the plate,
            # A1 at the top left, as in Harmony's plate view
            wy, wx = (row - 1) * pitch, (col - 1) * pitch
            well = _well_name(row, col)
            stack = f"r{row:02d}c{col:02d}f{field:02d}"
            data = TiffStackArray([p[2] for p in members], (len(members), ny, nx),
                                  dtype, folder=folder, transform=corrections[c])
            tiles.append(Tile(
                data=data,
                origin=(members[0][1], wy + sy * s["y"] - (ny - 1) / 2 * py, wx + sx * s["x"] - (nx - 1) / 2 * px),
                voxel_size=(dz, py, px),
                channel=names[c],
                timepoint=timepoint - 1,
                attrs={"stack": stack, "well": well, "row": row, "col": col, "opera_field": field,
                       "channel_id": c, "flatfield": corrections[c] is not None, "folder": str(folder)},
            ))
        if missing:
            logger.warning("%d plane(s) are marked as failed in %s: they are read as zeros.", missing, index.name)
        self._label_wells(tiles)
        return tiles

    @staticmethod
    def _label_wells(tiles: List[Tile]) -> None:
        """Fields of touching tiles within each well: ``A3_field0``, ``A3_field1``, ..."""
        wells: Dict[str, List[Tile]] = {}
        for t in tiles:
            wells.setdefault(t.attrs["well"], []).append(t)
        for well, members in wells.items():
            for t, k in zip(members, connected_components(members)):
                t.field = f"{well}_{FIELD_PREFIX}{k}"

    @staticmethod
    def _well_pitch(plate: Optional[Tuple[int, int]], stacks, tile_um: float) -> float:
        """Distance (um) between the centres of neighbouring wells.

        The pitch of the plate format (:data:`WELL_PITCH`), widened when the fields of a
        well (``tile_um`` wide tiles) would reach the next well; without a known format,
        the widest well plus 10%.
        """
        bounds: Dict[Tuple[int, int], List[float]] = {}
        for (row, col, *_), s in stacks.items():
            b = bounds.setdefault((row, col), [np.inf, -np.inf, np.inf, -np.inf])
            b[:] = min(b[0], s["y"]), max(b[1], s["y"]), min(b[2], s["x"]), max(b[3], s["x"])
        footprint = max(max(b[1] - b[0], b[3] - b[2]) for b in bounds.values()) + tile_um
        pitch = WELL_PITCH.get(plate)
        if pitch is None:
            if len(bounds) > 1:
                logger.info("Unknown plate format %s: wells are placed %.0f um apart.", plate, 1.1 * footprint)
            return 1.1 * footprint
        if pitch < footprint:
            logger.warning("The fields of a well span %.0f um, more than the %.0f um well pitch of a "
                           "%dx%d plate: wells are placed %.0f um apart.", footprint, pitch, *plate, 1.1 * footprint)
            return 1.1 * footprint
        return pitch

    @staticmethod
    def _channel_names(channels, used) -> Dict[int, str]:
        names = {c: (channels.get(c, {}).get("ChannelName", (None, None))[0] or f"ch{c}") for c in sorted(used)}
        counts: Dict[str, int] = {}
        for n in names.values():
            counts[n] = counts.get(n, 0) + 1
        return {c: n if counts[n] == 1 else f"{n} (ch{c})" for c, n in names.items()}

    @staticmethod
    def _flips(settings) -> Tuple[float, float]:
        """Signs of the ``y`` and ``x`` stage axes along image rows and columns."""
        m = _orientation(settings)
        if m is None:
            return 1.0, 1.0
        return (-1.0 if m[1][1] < 0 else 1.0), (-1.0 if m[0][0] < 0 else 1.0)

    @staticmethod
    def _pixel_sizes(channels, names) -> Tuple[float, float]:
        """``(y, x)`` pixel size in micrometres, calibrated by the ``OrientationMatrix``."""
        sizes = {}
        for c, name in names.items():
            s = channels.get(c, {})
            if "ImageResolutionX" not in s:
                raise ValueError(f"Channel {name}: no ImageResolutionX in the Opera index.")
            m = _orientation(s)
            mx, my = (abs(m[0][0]), abs(m[1][1])) if m else (1.0, 1.0)
            sizes[name] = (_micrometres(*s.get("ImageResolutionY", s["ImageResolutionX"])) * my,
                           _micrometres(*s["ImageResolutionX"]) * mx)
        values = list(sizes.values())
        if any(not np.allclose(v, values[0], rtol=1e-6) for v in values):
            raise ValueError(f"Opera channels have different pixel sizes {sizes}: they cannot be fused together.")
        return values[0]

    @staticmethod
    def _image_size(settings) -> Optional[Tuple[int, int]]:
        if "ImageSizeX" in settings and "ImageSizeY" in settings:
            return int(settings["ImageSizeY"][0]), int(settings["ImageSizeX"][0])
        return None

    @staticmethod
    def _header(folder: Path, name: str) -> Tuple[np.dtype, Tuple[int, int]]:
        """Data type and ``(y, x)`` shape of the planes, from the header of one file."""
        import tifffile

        with tifffile.TiffFile(os.path.join(folder, name)) as tif:
            page = tif.pages[0]
            return np.dtype(page.dtype), tuple(int(s) for s in page.shape[:2])
