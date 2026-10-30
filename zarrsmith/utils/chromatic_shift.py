"""Axial chromatic shift estimation and correction for multichannel volumes.

Light-sheet channels are often offset from each other along Z: the focal
planes of different wavelengths do not coincide, and the offset changes every
time the system is realigned. The offset is therefore estimated from the
sample itself:

1. A coarse pyramid level is segmented into a sample mask. Its surfaces that
   face along Z (its lower- and upper-Z sides) are the one structure every
   channel shares, through background and autofluorescence.
2. The Z-derivative of each channel is cross-correlated with the reference
   channel inside a thin band around those surfaces, giving a coarse shift.
3. Small crops centred on Z-facing surface points are read from a finer level,
   pre-shifted by the coarse estimate and correlated again. The correlation
   curves of all crops are summed and the peak, refined to sub-plane accuracy
   with a parabola fit, gives the final shift. Crops are balanced between
   the lower and upper surfaces so that channel-dependent edge blur, which
   biases the two in opposite directions, cancels out.

Lateral (YX-facing) edges are ignored on purpose: their profile is the same at
every Z, so they carry no information about an axial offset.

Shifts are corrections in the physical unit of the Z axis (usually
micrometers): the value for channel ``c`` is the displacement applied to that
channel, positive values moving its content towards higher Z indices. The
reference channel always gets ``0``.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any, Literal, Sequence

import dask.array as da
import numpy as np
import zarr
from scipy import ndimage as ndi

from .axes import normalize_axes, spatial_axes_in_order

logger = logging.getLogger("pymif")

Extent = Literal["union", "same", "intersection"]

# Root attribute recording the applied correction in the output store.
PROVENANCE_KEY = "pymif_chromatic_shift"

# Extra planes read around each crop so the Gaussian derivative has context.
_FILTER_MARGIN = 3


# ----------------------------------------------------------------------
# Axis and metadata helpers
# ----------------------------------------------------------------------

def _require_axes(axes: Sequence[str], required: str, label: str) -> None:
    missing = [ax for ax in required if ax not in axes]
    if missing:
        raise ValueError(
            f"{label} needs axes {required!r}, got {''.join(axes)!r} (missing {''.join(missing)!r})."
        )


def _zyx_spacings(metadata: dict[str, Any]) -> list[tuple[float, float, float]]:
    """Per-level ``(z, y, x)`` voxel spacing from the normalized metadata."""
    spatial = spatial_axes_in_order(normalize_axes(metadata.get("axes")))
    return [
        tuple(float(scale[spatial.index(ax)]) for ax in "zyx")
        for scale in metadata["scales"]
    ]


def _z_unit(metadata: dict[str, Any]) -> str | None:
    spatial = spatial_axes_in_order(normalize_axes(metadata.get("axes")))
    units = list(metadata.get("units") or [])
    if "z" in spatial and len(units) == len(spatial):
        return units[spatial.index("z")]
    return None


def _channel_volume(
    level: da.Array,
    axes: Sequence[str],
    timepoint: int,
    channel: int,
    region: tuple[slice, slice, slice] | None = None,
) -> da.Array:
    """Lazy ``zyx`` view of one timepoint and channel, optionally cropped."""
    zyx = dict(zip("zyx", region or (slice(None),) * 3))
    index = []
    for ax in axes:
        if ax == "t":
            index.append(timepoint)
        elif ax == "c":
            index.append(channel)
        else:
            index.append(zyx[ax])
    vol = level[tuple(index)]
    order = [ax for ax in axes if ax in "zyx"]
    return vol.transpose([order.index(ax) for ax in "zyx"])


def normalize_shifts(shifts: Any, n_channels: int) -> np.ndarray:
    """Accept a per-channel sequence or an :func:`estimate_chromatic_shift` result."""
    if isinstance(shifts, dict):
        if "shifts" not in shifts:
            raise ValueError("A shifts dict must contain a 'shifts' entry.")
        shifts = shifts["shifts"]
    out = np.asarray(shifts, dtype=float).ravel()
    if out.size != n_channels:
        raise ValueError(f"Expected one shift per channel ({n_channels}), got {out.size}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"Shifts must be finite numbers, got {out.tolist()}.")
    return out


# ----------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------

def _z_derivative(vol: np.ndarray) -> np.ndarray:
    """Gaussian-smoothed derivative along Z (axis 0)."""
    return ndi.gaussian_filter(vol.astype(np.float32, copy=False), sigma=1.0, order=(1, 0, 0), mode="nearest")


def _sample_mask(
    volumes: Sequence[np.ndarray],
    threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Foreground mask of the sample and the channel-averaged image it came from.

    Each channel is rescaled to roughly 0-1 (1st-99.5th percentile) before
    averaging, so no channel dominates. ``threshold`` applies to that
    averaged image; ``None`` uses Otsu. Only the largest connected component
    is kept, with holes filled.
    """
    combined = np.zeros(volumes[0].shape, np.float32)
    for vol in volumes:
        lo, hi = np.percentile(vol, (1.0, 99.5))
        combined += np.clip((vol - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    combined = ndi.gaussian_filter(combined / len(volumes), 1.0)

    if threshold is None:
        from skimage.filters import threshold_otsu

        threshold = float(threshold_otsu(combined))
    mask = combined > threshold

    labels, n = ndi.label(mask)
    if n > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == sizes.argmax()
    return ndi.binary_fill_holes(mask), combined


def _z_facing_surface(
    mask: np.ndarray,
    spacing: Sequence[float],
    min_z_normal: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Surface voxels of ``mask`` whose outward normal points mostly along Z.

    Returns the surface and the sign of the Z-gradient of the mask (``+1``
    where the sample lies above the surface, ``-1`` below). Faces created
    by the edge of the stack are not surfaces: the sample is just cut there.
    """
    eroded = ndi.binary_erosion(mask, border_value=1)
    surface = mask & ~eroded
    smooth = ndi.gaussian_filter(mask.astype(np.float32), 1.0)
    gz, gy, gx = np.gradient(smooth, *spacing)
    norm = np.sqrt(gz * gz + gy * gy + gx * gx)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_normal = np.abs(gz) / norm
    surface &= (norm > 0) & (z_normal >= min_z_normal)
    return surface, np.sign(gz)


def _ncc_curve(
    a: np.ndarray,
    b: np.ndarray,
    max_lag: int,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Normalized cross-correlation of ``a`` with Z-shifted windows of ``b``.

    ``b`` is ``2 * max_lag`` planes taller than ``a``; lag ``L`` compares
    ``a`` with ``b[max_lag + L : max_lag + L + len(a)]``, so a peak at ``L``
    means ``b`` is displaced by ``+L`` planes relative to ``a``. With a
    boolean ``weight`` only those voxels of ``a`` take part.
    """
    if weight is not None:
        zz, yy, xx = np.nonzero(weight)
        a_vals = a[zz, yy, xx]

        def window(lag):
            return b[zz + max_lag + lag, yy, xx]
    else:
        nz = a.shape[0]
        a_vals = a

        def window(lag):
            return b[max_lag + lag: max_lag + lag + nz]

    a_norm = math.sqrt(float(np.sum(a_vals * a_vals)))
    curve = np.zeros(2 * max_lag + 1)
    for i, lag in enumerate(range(-max_lag, max_lag + 1)):
        w = window(lag)
        denom = a_norm * math.sqrt(float(np.sum(w * w)))
        curve[i] = float(np.sum(a_vals * w)) / denom if denom > 0 else 0.0
    return curve


def _subpixel_peak(curve: np.ndarray) -> tuple[float, bool]:
    """Peak position of ``curve`` with a parabola fit, and whether it hit the edge."""
    i = int(np.argmax(curve))
    if i in (0, len(curve) - 1):
        return float(i), True
    y0, y1, y2 = curve[i - 1: i + 2]
    denom = y0 - 2 * y1 + y2
    offset = 0.5 * (y0 - y2) / denom if denom < 0 else 0.0
    return i + float(np.clip(offset, -0.5, 0.5)), False


def _spread_surface_points(
    surface: np.ndarray,
    facing: np.ndarray,
    strength: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Pick up to ``n_points`` strong surface voxels spread evenly over YX.

    The YX bounding box of the surface is split into a grid; each cell
    contributes its strongest lower-surface and its strongest upper-surface
    voxel, and the strongest of those are kept. Returns an ``(n, 3)`` array
    of ``zyx`` indices.
    """
    zz, yy, xx = np.nonzero(surface)
    if zz.size == 0 or n_points < 1:
        return np.empty((0, 3), dtype=int)

    cells = max(1, math.ceil(math.sqrt(n_points / 2)))

    def grid_index(v):
        extent = int(v.max() - v.min()) + 1
        return np.minimum((v - v.min()) * cells // extent, cells - 1)

    cell = grid_index(yy) * cells + grid_index(xx)
    key = cell * 2 + (facing[zz, yy, xx] > 0)
    s = strength[zz, yy, xx]

    order = np.lexsort((-s, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    picked = order[first]
    picked = picked[np.argsort(-s[picked], kind="stable")][:n_points]
    return np.stack([zz[picked], yy[picked], xx[picked]], axis=1)


def _auto_coarse_level(
    data_levels: Sequence[da.Array],
    axes: Sequence[str],
    min_planes: int = 16,
    min_yx: int = 32,
) -> int:
    """Coarsest level that still shows the sample's Z surfaces.

    The Z spacing of the level does not matter: the coarse estimate is
    sub-plane and refinement absorbs an error of about one coarse plane, so
    only enough planes and pixels to segment the sample are needed.
    """
    z_idx, y_idx, x_idx = (axes.index(ax) for ax in "zyx")
    for level in range(len(data_levels) - 1, -1, -1):
        shape = data_levels[level].shape
        if shape[z_idx] >= min_planes and min(shape[y_idx], shape[x_idx]) >= min_yx:
            return level
    return 0


def estimate_chromatic_shift(
    data_levels: Sequence[da.Array],
    metadata: dict[str, Any],
    *,
    reference_channel: int = 0,
    timepoint: int = 0,
    coarse_level: int | None = None,
    refine_level: int | None = 0,
    max_shift: float | None = None,
    n_regions: int = 32,
    crop_yx: int = 64,
    min_z_normal: float = 0.7,
    min_correlation: float = 0.2,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Estimate the axial (Z) shift of every channel relative to a reference.

    Parameters
    ----------
    data_levels, metadata
        Multiscale pyramid (finest first) and normalized PyMIF metadata. The
        axes must include ``c``, ``z``, ``y`` and ``x``.
    reference_channel
        Channel that stays in place; the others are aligned to it.
    timepoint
        Timepoint used for the estimate when there is a ``t`` axis.
    coarse_level
        Level used for the sample mask and the coarse estimate. ``None``
        picks the coarsest level with at least 16 Z planes and 32 pixels in
        Y and X.
    refine_level
        Level the surface crops are read from. ``None`` skips refinement
        and returns the coarse estimate.
    max_shift
        Largest shift searched, in the physical unit of the Z axis. ``None``
        searches up to 10% of the stack depth. Searching wide is cheap: only
        the coarse level is affected.
    n_regions
        Number of surface crops used for refinement.
    crop_yx
        Y and X size of each crop, in pixels of ``refine_level``.
    min_z_normal
        Minimum ``|n_z|`` of the surface normal for a surface voxel to count
        as Z-facing (``1`` = exactly horizontal surface).
    min_correlation
        Crops whose correlation peak is below this are left out.
    threshold
        Mask threshold on the channel-averaged, 0-1 rescaled coarse image.
        ``None`` uses Otsu.

    Returns
    -------
    dict
        JSON-serializable report. ``shifts`` holds the correction per channel
        (pass the whole dict to :func:`write_chromatic_shift_corrected`).
        ``surface_shifts`` gives the median per-crop shift on the lower and
        upper sample surface separately; a large difference between the two
        points at channel-dependent edge blur rather than a pure offset.
        ``spread`` is a robust standard deviation of the per-crop shifts
        around their surface median, and ``regions`` lists every crop with
        its position, shift and correlation so position-dependent shifts can
        be spotted. ``warnings`` collects the warnings raised during the
        estimate (search limit reached, no usable crop...), so pipelines can
        fail on them without catching Python warnings.
    """
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")
    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    _require_axes(axes, "czyx", "Chromatic shift estimation")

    n_levels = len(data_levels)
    n_channels = int(data_levels[0].shape[axes.index("c")])
    if not 0 <= reference_channel < n_channels:
        raise ValueError(f"reference_channel={reference_channel} is out of range for {n_channels} channels.")
    if "t" in axes and not 0 <= timepoint < data_levels[0].shape[axes.index("t")]:
        raise ValueError(f"timepoint={timepoint} is out of range.")
    spacings = _zyx_spacings(metadata)
    if max_shift is None:
        max_shift = 0.1 * data_levels[0].shape[axes.index("z")] * spacings[0][0]
    if max_shift <= 0:
        raise ValueError("max_shift must be positive.")
    unit = _z_unit(metadata)
    if coarse_level is None:
        coarse_level = _auto_coarse_level(data_levels, axes)
    if not 0 <= coarse_level < n_levels:
        raise ValueError(f"coarse_level={coarse_level} is out of range for {n_levels} levels.")
    if refine_level is not None and not 0 <= refine_level <= coarse_level:
        raise ValueError(
            f"refine_level={refine_level} must be between 0 and coarse_level={coarse_level}."
        )

    report: dict[str, Any] = {
        "shifts": [0.0] * n_channels,
        "unit": unit,
        "reference_channel": int(reference_channel),
        "timepoint": int(timepoint) if "t" in axes else None,
        "channel_names": list(metadata.get("channel_names") or []),
        "coarse_level": int(coarse_level),
        "refine_level": None if refine_level is None else int(refine_level),
        "coarse_shifts": [0.0] * n_channels,
        "coarse_correlation": [1.0 if c == reference_channel else None for c in range(n_channels)],
        "spread": [0.0 if c == reference_channel else None for c in range(n_channels)],
        "n_regions_used": [0] * n_channels,
        "surface_shifts": [None] * n_channels,
        "max_shift": float(max_shift),
        "warnings": [],
        "regions": [],
    }
    if n_channels == 1:
        return report

    def warn(message: str) -> None:
        report["warnings"].append(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    others = [c for c in range(n_channels) if c != reference_channel]

    # -- coarse estimate on the whole Z-facing surface --------------------
    coarse = data_levels[coarse_level]
    sp_c = spacings[coarse_level]
    vols = da.compute(*[_channel_volume(coarse, axes, timepoint, c) for c in range(n_channels)])
    vols = [np.asarray(v, dtype=np.float32) for v in vols]

    mask, combined = _sample_mask(vols, threshold)
    surface, facing = _z_facing_surface(mask, sp_c, min_z_normal)
    if not surface.any():
        raise ValueError(
            f"No Z-facing sample surface found at level {coarse_level}. The sample may fill "
            "the whole stack in Z, or the mask threshold may be off (try `threshold`, "
            "`coarse_level` or a lower `min_z_normal`)."
        )
    band = ndi.binary_dilation(surface, iterations=2)

    # At least ±2 planes, so a sub-plane shift still has neighbours for the peak fit.
    max_lag = max(2, math.ceil(max_shift / sp_c[0]))
    deriv = [_z_derivative(v) for v in vols]
    pad = ((max_lag, max_lag), (0, 0), (0, 0))

    displacement = np.zeros(n_channels)
    for c in others:
        curve = _ncc_curve(deriv[reference_channel], np.pad(deriv[c], pad), max_lag, weight=band)
        peak, at_edge = _subpixel_peak(curve)
        if at_edge:
            warn(
                f"Channel {c}: the coarse shift reached the search limit of ±{max_lag * sp_c[0]:g} "
                f"{unit or 'units'}; increase max_shift."
            )
        displacement[c] = (peak - max_lag) * sp_c[0]
        report["coarse_correlation"][c] = float(curve.max())
    report["coarse_shifts"] = [float(-d) for d in displacement]

    # -- refinement on crops around Z-facing surface points ---------------
    if refine_level is not None:
        fine = data_levels[refine_level]
        sp_f = spacings[refine_level]
        shape_f = tuple(int(fine.shape[axes.index(ax)]) for ax in "zyx")
        ratio = sp_c[0] / sp_f[0]
        search = math.ceil(ratio) + 2     # residual search range, planes
        half_z = math.ceil(ratio) + 6     # reference window half height, planes
        half_yx = max(4, crop_yx // 2)
        m = _FILTER_MARGIN

        strength = np.abs(ndi.gaussian_filter(combined, 1.0, order=(1, 0, 0)))
        points = _spread_surface_points(surface, facing, strength, n_regions)
        centers = np.rint(points * np.asarray(sp_c) / np.asarray(sp_f)).astype(int)
        pre_shift = {c: int(round(displacement[c] / sp_f[0])) for c in others}

        def bounds(center, half, size):
            return max(0, center - half), min(size, center + half)

        # Crops reaching past either end of the stack are read up to
        # the edge and padded with the edge plane: samples are often imaged
        # with little margin, and those crops still hold the surface.
        jobs = []   # (region index, channel, Z edge padding) per lazy read
        reads = []

        def add_read(r, c, z0, z1, ys, xs):
            lo, hi = max(0, z0), min(shape_f[0], z1)
            if hi - lo < (z1 - z0) // 2:
                return False
            jobs.append((r, c, (lo - z0, z1 - hi)))
            reads.append(_channel_volume(fine, axes, timepoint, c, (slice(lo, hi), ys, xs)))
            return True

        for r, (zc, yc, xc) in enumerate(centers):
            ys = slice(*bounds(yc, half_yx + m, shape_f[1]))
            xs = slice(*bounds(xc, half_yx + m, shape_f[2]))
            if ys.stop - ys.start <= 2 * m or xs.stop - xs.start <= 2 * m:
                continue
            z0 = zc - half_z - m
            z1 = zc + half_z + m + 1
            if not add_read(r, reference_channel, z0, z1, ys, xs):
                continue
            for c in others:
                add_read(r, c, z0 + pre_shift[c] - search, z1 + pre_shift[c] + search, ys, xs)

        crops = da.compute(*reads) if reads else ()
        ref_deriv = {}    # region -> derivative of the reference crop
        chan_deriv = {}   # (region, channel) -> derivative of the channel crop
        for (r, c, z_pad), crop in zip(jobs, crops):
            crop = np.pad(np.asarray(crop, dtype=np.float32), (z_pad, (0, 0), (0, 0)), mode="edge")
            d = _z_derivative(crop)[m:-m, m:-m, m:-m]
            if c == reference_channel:
                ref_deriv[r] = d
            else:
                chan_deriv[(r, c)] = d

        for c in others:
            total = np.zeros(2 * search + 1)
            used = {"lower": [], "upper": []}   # per-crop corrections by surface
            for r in sorted(ref_deriv):
                if (r, c) not in chan_deriv:
                    continue
                curve = _ncc_curve(ref_deriv[r], chan_deriv[(r, c)], search)
                peak, _ = _subpixel_peak(curve)
                corr = float(curve.max())
                ok = corr >= min_correlation
                # The mask rises along Z on the sample's lower-Z surface.
                side = "lower" if facing[tuple(points[r])] > 0 else "upper"
                region_shift = -(pre_shift[c] + peak - search) * sp_f[0]
                report["regions"].append({
                    "channel": int(c),
                    "position": [float(v) for v in centers[r] * np.asarray(sp_f)],
                    "surface": side,
                    "shift": float(region_shift),
                    "correlation": corr,
                    "used": bool(ok),
                })
                if ok:
                    total += curve
                    used[side].append(region_shift)

            report["surface_shifts"][c] = {
                side: float(np.median(v)) if v else None for side, v in used.items()
            }
            # Spread around each surface's own median, so a systematic
            # lower/upper difference (see surface_shifts) does not inflate it.
            residuals = np.concatenate([np.asarray(v) - np.median(v) for v in used.values() if v] or [[]])
            report["n_regions_used"][c] = int(residuals.size)
            if residuals.size > 1:
                report["spread"][c] = float(1.4826 * np.median(np.abs(residuals)))
            if not residuals.size:
                warn(
                    f"Channel {c}: no surface crop reached min_correlation={min_correlation}; "
                    "keeping the coarse estimate."
                )
                continue
            peak, at_edge = _subpixel_peak(total)
            if at_edge:
                warn(
                    f"Channel {c}: the refined shift sits at the edge of the refinement window; "
                    "the coarse estimate may be off."
                )
            displacement[c] = (pre_shift[c] + peak - search) * sp_f[0]

    report["shifts"] = [0.0 if c == reference_channel else float(-displacement[c]) for c in range(n_channels)]
    for c in others:
        logger.info(
            "Chromatic shift channel %d: %.3f %s (coarse %.3f, %d regions, spread %s)",
            c, report["shifts"][c], unit or "", report["coarse_shifts"][c],
            report["n_regions_used"][c], report["spread"][c],
        )
    return report


# ----------------------------------------------------------------------
# Correction
# ----------------------------------------------------------------------

def _ceil_to_multiple(value: float, step: int) -> int:
    return int(math.ceil(value / step - 1e-9)) * step


def plan_z_extent(
    nz0: int,
    z_spacings: Sequence[float],
    shifts: np.ndarray,
    extent: Extent,
) -> tuple[float, float]:
    """Z padding of the output in physical units, as ``(below, total)``.

    ``below`` is the padding added before plane 0 (negative when cropping),
    ``total`` the change in Z extent. Both are whole multiples of the
    coarsest level's Z spacing, so every pyramid level is padded by a whole
    number of planes and the levels stay aligned.

    - ``"union"``: grow Z so no channel loses data.
    - ``"same"``: keep the input size; data shifted out of the stack is lost.
    - ``"intersection"``: crop Z to the planes every channel still covers.
    """
    sz0 = float(z_spacings[0])
    ratios = [s / sz0 for s in z_spacings]
    if all(abs(r - round(r)) < 1e-3 for r in ratios):
        step = math.lcm(*[max(1, int(round(r))) for r in ratios])
    else:
        warnings.warn(
            "Pyramid levels do not have integer Z spacing ratios; coarse levels may be "
            "misaligned by up to one plane after the correction.",
            UserWarning,
            stacklevel=3,
        )
        step = 1

    s = shifts / sz0
    if extent == "union":
        lo = _ceil_to_multiple(max(0.0, -s.min()), step)
        hi = _ceil_to_multiple(max(0.0, s.max()), step)
    elif extent == "intersection":
        lo = -_ceil_to_multiple(max(0.0, s.max()), step)
        hi = -_ceil_to_multiple(max(0.0, -s.min()), step)
    elif extent == "same":
        lo = hi = 0
    else:
        raise ValueError(f"extent must be 'union', 'same' or 'intersection', got {extent!r}.")

    if nz0 + lo + hi <= 0:
        raise ValueError(
            f"extent={extent!r} leaves no Z planes: the shifts ({shifts.tolist()}) exceed the stack depth."
        )
    return lo * sz0, (lo + hi) * sz0


def _cast_like(arr: da.Array, dtype: np.dtype) -> da.Array:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return da.clip(da.round(arr), info.min, info.max).astype(dtype)
    return arr.astype(dtype)


def shift_along_axis(
    arr: da.Array,
    axis: int,
    n_out: int,
    offset: float,
    order: int = 1,
) -> da.Array:
    """Return ``out[j] = arr[j - offset]`` along ``axis``, length ``n_out``.

    Positions outside the input are filled with zeros. ``order=1`` uses
    linear interpolation for fractional offsets; ``order=0`` rounds the
    offset to whole planes, which copies the data without changing values.
    """
    if order not in (0, 1):
        raise ValueError(f"order must be 0 or 1, got {order}.")
    if order == 0:
        offset = float(round(offset))
    n = math.floor(offset)
    frac = offset - n
    if frac < 1e-6:
        frac = 0.0
    elif frac > 1 - 1e-6:
        n, frac = n + 1, 0.0

    nz = arr.shape[axis]
    first = -n - (1 if frac else 0)   # lowest source index needed
    last = n_out - 1 - n              # highest source index needed
    left = max(0, -first)
    right = max(0, last - (nz - 1))
    if left or right:
        pad = [(0, 0)] * arr.ndim
        pad[axis] = (left, right)
        arr = da.pad(arr, pad, mode="constant", constant_values=0)

    def take(start):
        index = [slice(None)] * arr.ndim
        index[axis] = slice(start + left, start + left + n_out)
        return arr[tuple(index)]

    current = take(-n)
    if not frac:
        return current
    work = np.result_type(arr.dtype, np.float32)
    below = take(-n - 1)
    out = (1 - frac) * current.astype(work) + frac * below.astype(work)
    return _cast_like(out, arr.dtype)


def shift_levels(
    data_levels: Sequence[da.Array],
    metadata: dict[str, Any],
    shifts: np.ndarray | None,
    pad_below: float,
    pad_total: float,
    order: int = 1,
) -> list[da.Array]:
    """Lazily shift every channel of every level along Z and resize Z.

    ``shifts`` (one per channel, physical units) is ``None`` for data
    without a channel axis such as labels: those are only padded or cropped
    to the new extent, always without interpolation.
    """
    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    z_axis = axes.index("z")
    c_axis = axes.index("c") if "c" in axes else None
    spacings = [sp[0] for sp in _zyx_spacings(metadata)]

    out = []
    for level, sz in zip(data_levels, spacings):
        below = int(round(pad_below / sz))
        n_out = int(level.shape[z_axis]) + int(round(pad_total / sz))
        if shifts is None or c_axis is None:
            out.append(shift_along_axis(level, z_axis, n_out, below, order=0))
            continue
        channels = []
        for c, shift in enumerate(shifts):
            index = [slice(None)] * level.ndim
            index[c_axis] = slice(c, c + 1)
            channels.append(
                shift_along_axis(level[tuple(index)], z_axis, n_out, below + shift / sz, order=order)
            )
        out.append(da.concatenate(channels, axis=c_axis))
    return out


def _mirrored_array_kwargs(src: zarr.Array, shape: Sequence[int]) -> dict[str, Any]:
    """``create_array`` keywords reproducing ``src`` (codecs, chunks, shards...) at a new shape."""
    kwargs: dict[str, Any] = {
        "name": src.basename,
        "shape": tuple(int(s) for s in shape),
        "dtype": src.dtype,
        "chunks": src.chunks,
        "compressors": src.compressors,
        "filters": src.filters,
        "fill_value": src.fill_value,
    }
    if src.metadata.zarr_format == 3:
        kwargs.update(
            shards=src.shards,
            serializer=src.serializer,
            chunk_key_encoding=src.metadata.chunk_key_encoding,
            dimension_names=src.metadata.dimension_names,
        )
    else:
        kwargs.update(
            order=src.order,
            chunk_key_encoding={"name": "v2", "separator": src.metadata.dimension_separator},
        )
    return kwargs


def write_chromatic_shift_corrected(
    source_root: zarr.Group,
    datasets: Sequence[tuple[str, Any]],
    path: str | Path,
    shifts: Any,
    *,
    extent: Extent = "union",
    order: int = 1,
    overwrite: bool = True,
    num_workers: int | None = None,
    progress: bool = True,
) -> zarr.Group:
    """Write a Z-shift-corrected copy of an OME-Zarr store.

    ``datasets`` are ``(name, dataset)`` pairs as yielded by
    ``ZarrManager._iter_datasets``, the raw image first; each dataset needs
    ``data``, ``zarr_data`` (the source arrays), ``metadata`` and ``path``.
    Every output array copies its source's chunks, shards, codecs, fill value
    and dimension names, and every group copies its source attributes, so
    the output has the structure of the input with only the Z size changed.
    The applied correction is recorded in the root attribute
    ``pymif_chromatic_shift``.

    Image datasets whose channel count matches ``shifts`` are corrected
    channel by channel. Datasets without a channel axis (labels) follow the
    reference channel: they are only padded or cropped to the new Z extent.
    Image datasets with a different channel count are skipped with a warning.
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("No datasets to write.")
    raw_name, raw = datasets[0]
    raw_axes = normalize_axes(raw.metadata.get("axes"), ndim=raw.data[0].ndim)
    _require_axes(raw_axes, "czyx", "Chromatic shift correction")
    n_channels = int(raw.data[0].shape[raw_axes.index("c")])
    report = shifts if isinstance(shifts, dict) else None
    shifts = normalize_shifts(shifts, n_channels)

    raw_z = [sp[0] for sp in _zyx_spacings(raw.metadata)]
    pad_below, pad_total = plan_z_extent(
        int(raw.data[0].shape[raw_axes.index("z")]), raw_z, shifts, extent
    )

    from .ngff import ZarrWriteConfig, _run_stores

    cfg = ZarrWriteConfig(num_workers=num_workers, progress=progress)
    root = zarr.open_group(
        str(Path(path)),
        mode="w" if overwrite else "w-",
        zarr_format=source_root.metadata.zarr_format,
    )
    root.attrs.update(source_root.attrs.asdict())
    if any(name.startswith("labels/") for name, _ in datasets) and "labels" in source_root:
        root.require_group("labels").attrs.update(source_root["labels"].attrs.asdict())

    for name, dataset in datasets:
        if not dataset.data:
            continue
        if dataset.zarr_data is None or len(dataset.zarr_data) != len(dataset.data):
            raise ValueError(
                f"Dataset {name!r} has in-memory changes that are not on disk, so its "
                "on-disk structure is unknown. Write it with to_zarr() first."
            )
        axes = normalize_axes(dataset.metadata.get("axes"), ndim=dataset.data[0].ndim)
        if "z" not in axes:
            warnings.warn(f"Skipping dataset {name!r}: it has no 'z' axis.", UserWarning, stacklevel=2)
            continue
        dataset_shifts = shifts
        if "c" not in axes:
            dataset_shifts = None
        elif dataset.data[0].shape[axes.index("c")] != n_channels:
            warnings.warn(
                f"Skipping dataset {name!r}: it has {dataset.data[0].shape[axes.index('c')]} "
                f"channels but {n_channels} shifts were given.",
                UserWarning,
                stacklevel=2,
            )
            continue

        levels = shift_levels(dataset.data, dataset.metadata, dataset_shifts, pad_below, pad_total, order=order)

        group_path = (dataset.path or "/").strip("/")
        group = root if not group_path else root.require_group(group_path)
        if group_path:
            group.attrs.update(source_root[group_path].attrs.asdict())

        tasks = []
        block_bytes = 0
        for arr, src in zip(levels, dataset.zarr_data):
            dst = group.create_array(**_mirrored_array_kwargs(src, arr.shape))
            dst.attrs.update(src.attrs.asdict())
            block = src.shards or src.chunks
            if src.shards is not None:
                block_bytes = max(block_bytes, math.prod(block) * np.dtype(src.dtype).itemsize)
            tasks.append(da.store(arr.rechunk(block), dst, lock=False, compute=False))
        _run_stores(tasks, cfg, block_bytes=block_bytes, desc=name)

    provenance: dict[str, Any] = {
        "shifts": [float(s) for s in shifts],
        "unit": _z_unit(raw.metadata),
        "extent": extent,
        "order": int(order),
        # Where input plane 0 of the reference channel sits in the output.
        "z_offset": float(pad_below),
    }
    if report is not None:
        provenance["estimate"] = {
            key: report[key]
            for key in ("reference_channel", "timepoint", "coarse_level", "refine_level",
                        "coarse_shifts", "spread", "n_regions_used")
            if key in report
        }
    root.attrs[PROVENANCE_KEY] = provenance
    logger.info("Chromatic shift corrected dataset written to %s (shifts %s).", path, provenance["shifts"])
    return root
