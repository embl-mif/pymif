"""napari widget to inspect raw mosaic tiles and fuse them into an OME-Zarr image."""

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from magicgui import magicgui
from napari import current_viewer
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget

from pymif import mosaic_manager as mos
from ._batch_csv import append_batch_csv
from ._layout import compose_dock, enable_drag_and_drop, make_button_holder, make_section
from ._log import EmittingStream, make_log_appender

#: Mosaic readers offered in the widget (Opera and Zeiss are placeholders for now).
MOSAIC_READERS = {
    "luxendo": mos.LuxendoMosaicManager,
    "opera": mos.OperaMosaicManager,
    "zeiss": mos.ZeissMosaicManager,
}

#: Label used in the widgets for tiles without a group.
NO_GROUP = "(none)"

_PROJECTIONS = {"none (3D)": None, "max": "max", "sum": "sum"}

#: Which tiles the fusion reads; the others are left blank (see ``skipped_tile_ids``).
TILE_MODES = {"all tiles": None, "visible layers only": "visible", "selected layers only": "selected"}

#: Name of the Shapes layer with one rectangle per stack position, clicked to pick tiles.
TILE_MAP = "tile map"

#: Face and edge colours (RGBA) of picked and unpicked rectangles on the tile map.
_MAP_COLORS = {True: ((0.2, 0.9, 0.3, 0.15), (0.2, 0.9, 0.3, 1.0)),
               False: ((0.5, 0.5, 0.5, 0.0), (0.5, 0.5, 0.5, 0.6))}

#: A press that moves more than this (canvas pixels) is a pan, not a click on the tile map.
_CLICK_TOLERANCE = 4

#: Columns of the CSV read by ``pymif fuse --batch``.
MOSAIC_BATCH_COLUMNS = [
    "input", "microscope", "output",
    "downsample", "projection", "blend", "channels", "groups", "tiles", "skip_tiles",
    "midline_axis", "out_dtype", "read_workers", "temp_dir", "overwrite",
    "chunk_size", "max_size(MB)", "scene_index", "zarr_format", "downscale_factor", "subset",
    "channel_colors", "channel_names", "num_levels", "shards", "shard_target_mb", "shard_exclude_axes",
    "drop_singleton", "num_workers",
]

_TITLE_STYLE = "QLabel { font-weight: bold; font-size: 13px; padding: 2px 0px; }"


# ---------------------------------------------------------------------------
# helpers (no Qt, unit tested)
# ---------------------------------------------------------------------------

def group_labels(mosaic) -> list[str]:
    """Group names of a mosaic as shown in the widget."""
    return [NO_GROUP if g is None else str(g) for g in mosaic.groups]


def select_tiles(mosaic, channels: Sequence[str], groups: Sequence[str]):
    """Tiles whose channel is in ``channels`` and group label is in ``groups``."""
    channels, groups = set(channels), set(groups)
    return [
        t for t in mosaic.tiles
        if t.channel in channels and (NO_GROUP if t.group is None else str(t.group)) in groups
    ]


def default_output(input_path: Path, projection: Optional[str], ds: Sequence[int]) -> Path:
    """``<folder>/<name>_MIP_ds8-8.ome.zarr`` (projection) or ``<name>_3D_ds4-8-8.ome.zarr``."""
    input_path = Path(input_path)
    if projection is None:
        kind = "3D_ds" + "-".join(str(v) for v in ds)
    else:
        kind = ("MIP" if projection == "max" else "SIP") + "_ds" + "-".join(str(v) for v in ds[1:])
    return input_path / f"{input_path.name}_{kind}.ome.zarr"


def folder_of(path: Path) -> Path:
    """Dropped on the acquisition field: a file stands for its folder."""
    path = Path(path)
    return path.parent if path.is_file() else path


def output_for_drop(path: Path, current: Path) -> Path:
    """Dropped on the output field: a ``.zarr`` is used as is, a plain folder
    receives the current output name (``fused.ome.zarr`` if there is none)."""
    path = Path(path)
    if path.suffix == ".zarr":
        return path
    folder = path if path.is_dir() else path.parent
    name = Path(current).name if Path(current).suffix == ".zarr" else "fused.ome.zarr"
    return folder / name


def fuse_kwargs(
    *,
    channels: Sequence[str],
    projection: str,
    blend: str,
    ds_z: int,
    ds_y: int,
    ds_x: int,
    levels: int,
    workers: int,
    overwrite: bool,
) -> dict:
    """Map the fusion widget values onto :func:`~pymif.mosaic_manager.fuse_tiles` arguments."""
    return {
        "channels": list(channels),
        "projection": _PROJECTIONS[projection],
        "blend": blend,
        "ds": (int(ds_z), int(ds_y), int(ds_x)),
        "levels": int(levels),
        "workers": int(workers),
        "overwrite": bool(overwrite),
    }


def tile_layers(layers, mosaic_path) -> list:
    """The napari layers that show a tile of the mosaic loaded from ``mosaic_path``."""
    return [
        layer for layer in layers
        if (getattr(layer, "metadata", None) or {}).get("mosaic_path") == str(mosaic_path)
        and "mosaic_tile_id" in layer.metadata
    ]


def skipped_tile_ids(layers, selection, mosaic_path, mode: Optional[str], all_channels: bool = False) -> list[str]:
    """Ids of the tiles to leave blank, from the tile layers shown in napari.

    ``mode="visible"`` skips the tiles whose layer is hidden (eye icon off),
    ``mode="selected"`` the tiles whose layer is not selected in the layer list,
    ``None`` none. Tiles without a layer (e.g. a channel that is not shown) are
    fused normally. With ``all_channels`` the same stack position is skipped in
    every channel (the ids are then stack names, e.g. ``"1-x00-y00"``).
    """
    if mode is None:
        return []
    ids = []
    for layer in tile_layers(layers, mosaic_path):
        keep = layer.visible if mode == "visible" else layer in selection
        if not keep:
            ids.append(layer.metadata["mosaic_tile_id"])
    if all_channels:
        ids = list(dict.fromkeys(i.split("/", 1)[1] for i in ids))
    return ids


def stack_id(mosaic, tile) -> str:
    """Stack position of a tile, shared by all its channels (the ``<stack>`` of its tile id)."""
    return mosaic.tile_id(tile).split("/", 1)[1]


def tile_footprints(mosaic) -> dict:
    """``{stack id: [[y0, x0], [y1, x1]]}``: the y/x rectangle covered by every stack
    position (all its channels and sides), in sample coordinates."""
    out = {}
    for t in mosaic.tiles:
        key = stack_id(mosaic, t)
        lo, hi = np.array(t.extent[0][1:]), np.array(t.extent[1][1:])
        if key in out:
            lo, hi = np.minimum(out[key][0], lo), np.maximum(out[key][1], hi)
        out[key] = np.array([lo, hi])
    return out


def add_tile_map(viewer, mosaic):
    """Add the tile map of ``mosaic`` to ``viewer`` (replacing any earlier one), all tiles picked.

    One rectangle per stack position, labelled with its stack id. The layer cannot
    be edited (napari would let you move or delete rectangles); :func:`toggle_tile_at`
    picks or unpicks a tile.
    """
    from pymif.utils.visualize import units_for_axes

    for layer in [layer for layer in viewer.layers if layer.metadata.get("mosaic_tile_map")]:
        viewer.layers.remove(layer)
    footprints = tile_footprints(mosaic)
    size = np.median([np.min(r[1] - r[0]) for r in footprints.values()])
    # the same units as the tile layers, or napari ignores units for every layer
    units = units_for_axes({"axes": "zyx", "units": (mosaic.tiles[0].units,) * 3}, "yx")
    layer = viewer.add_shapes(
        list(footprints.values()),
        shape_type="rectangle",
        name=TILE_MAP,
        features={"stack": list(footprints), "picked": [True] * len(footprints)},
        text={"string": "{stack}", "size": 8, "color": "white"},
        edge_width=max(size / 100, 1e-6),
        metadata={"mosaic_tile_map": True, "mosaic_path": str(mosaic.path)},
        **({} if units is None else {"units": units}),
    )
    layer.editable = False
    # napari makes shapes editable again when the display switches between 2D and 3D
    layer.events.editable.connect(lambda _: setattr(layer, "editable", False) if layer.editable else None)
    refresh_tile_map(layer)
    return layer


def find_tile_map(layers, mosaic_path):
    """The tile map layer of the mosaic loaded from ``mosaic_path``, or ``None``."""
    for layer in layers:
        md = getattr(layer, "metadata", None) or {}
        if md.get("mosaic_tile_map") and md.get("mosaic_path") == str(mosaic_path):
            return layer
    return None


def refresh_tile_map(layer, picked: Optional[Sequence[bool]] = None) -> None:
    """Colour the rectangles of a tile map by their ``picked`` feature, or by ``picked``
    if given (a preview that leaves the feature unchanged)."""
    picked = list(layer.features["picked"]) if picked is None else list(picked)
    layer.face_color = np.array([_MAP_COLORS[bool(p)][0] for p in picked])
    layer.edge_color = np.array([_MAP_COLORS[bool(p)][1] for p in picked])


def tile_at(layer, position) -> Optional[int]:
    """Index of the tile-map rectangle under ``position`` (world coordinates), or ``None``."""
    index, _ = layer.get_value(position, world=True) or (None, None)
    return index


def _box(corner, opposite):
    """``(lo, hi)`` y/x corners of the box between two world positions."""
    corner, opposite = np.asarray(corner)[-2:], np.asarray(opposite)[-2:]
    return np.minimum(corner, opposite), np.maximum(corner, opposite)


def tiles_in_box(layer, corner, opposite) -> list[int]:
    """Indices of the tile-map rectangles that lie, even partially, in the y/x box
    between two world positions."""
    lo, hi = _box(corner, opposite)
    rect_lo = np.array([np.min(r, axis=0) for r in layer.data])
    rect_hi = np.array([np.max(r, axis=0) for r in layer.data])
    return np.flatnonzero(np.all((rect_lo <= hi) & (rect_hi >= lo), axis=1)).tolist()


def show_drag_box(viewer, tile_map, corner=None, opposite=None) -> None:
    """Draw the box between two world positions over ``tile_map``, or remove it (no positions).

    The box is a temporary shapes layer (``"tile map box"``) that only exists while
    dragging. Adding it would select it, so the layer selection is restored at once
    (it decides which tiles are fused in *selected layers only* mode).
    """
    box_layer = next((layer for layer in viewer.layers if layer.metadata.get("mosaic_drag_box")), None)
    if corner is None:
        if box_layer is not None:
            viewer.layers.remove(box_layer)
        return
    rect = np.array(_box(corner, opposite))
    if box_layer is not None:
        box_layer.data = [rect]
        return
    selection, active = set(viewer.layers.selection), viewer.layers.selection.active
    viewer.add_shapes(
        [rect],
        shape_type="rectangle",
        name=f"{TILE_MAP} box",
        edge_color="white",
        face_color=[1.0, 1.0, 1.0, 0.1],
        edge_width=2 * float(np.max(tile_map.edge_width)),
        units=tile_map.units,
        metadata={"mosaic_drag_box": True},
    )
    viewer.layers.selection.clear()
    viewer.layers.selection.update(selection)
    viewer.layers.selection.active = active


def boxed_picks(layer, indices: Sequence[int], picked: bool) -> list[bool]:
    """The ``picked`` feature of a tile map with ``indices`` set to ``picked``."""
    out = [bool(p) for p in layer.features["picked"]]
    for i in indices:
        out[i] = picked
    return out


def set_picked(layer, indices: Sequence[int], picked: bool) -> None:
    """Pick (or unpick) the tile-map rectangles ``indices``."""
    features = layer.features.copy()
    features["picked"] = boxed_picks(layer, indices, picked)
    layer.features = features
    refresh_tile_map(layer)


def toggle_tile_at(layer, position) -> Optional[str]:
    """Pick or unpick the tile under ``position`` (world coordinates) on a tile map.

    Returns the stack id of the toggled tile, or ``None`` if no tile is there.
    """
    index = tile_at(layer, position)
    if index is None:
        return None
    set_picked(layer, [index], not bool(layer.features["picked"][index]))
    return str(layer.features["stack"][index])


def picked_stack_ids(layer) -> Optional[list[str]]:
    """Stack ids picked on a tile map; ``None`` (every tile) without a map."""
    if layer is None:
        return None
    return [str(s) for s, p in zip(layer.features["stack"], layer.features["picked"]) if p]


def picked_tiles(mosaic, tiles, picked: Optional[Sequence[str]]):
    """The ``tiles`` whose stack position is in ``picked`` (all of them if ``picked`` is ``None``)."""
    if picked is None:
        return list(tiles)
    picked = set(picked)
    return [t for t in tiles if stack_id(mosaic, t) in picked]


def batch_row(*, input_path, microscope: str, output, values: dict, all_groups: Sequence[str],
              skip_ids: Sequence[str], picked: Optional[Sequence[str]] = None,
              all_stacks: Sequence[str] = ()) -> dict:
    """One ``pymif fuse --batch`` CSV row from the fusion widget ``values``.

    ``picked`` are the stack ids picked on the tile map (``None``: all); the ``tiles``
    column stays empty (every tile) when they cover ``all_stacks``.
    """
    groups = list(values["groups"])
    return {
        "input": str(Path(input_path)),
        "microscope": microscope,
        "output": str(Path(output)),
        "downsample": f"{values['ds_z']} {values['ds_y']} {values['ds_x']}",
        "projection": _PROJECTIONS[values["projection"]] or "none",
        "blend": values["blend"],
        "channels": " ".join(values["channels"]),
        # every group selected = the default (all); "(none)" is how the CLI sees a missing group
        "groups": "" if set(groups) == set(all_groups) else " ".join("None" if g == NO_GROUP else g for g in groups),
        "tiles": "" if picked is None or set(picked) >= set(all_stacks) else " ".join(picked),
        "skip_tiles": " ".join(skip_ids),
        "num_levels": str(values["levels"]),
        "read_workers": str(values["workers"]),
        "overwrite": str(bool(values["overwrite"])).lower(),
    }


def shown_contrast_limits(layers, mosaic) -> dict:
    """``{channel: (low, high)}`` of the tile layers of ``mosaic`` shown in napari
    (their contrast is linked per channel), to keep it when tiles are shown again."""
    return {mosaic.tiles[layer.metadata["mosaic_tile"]].channel: tuple(float(v) for v in layer.contrast_limits)
            for layer in tile_layers(layers, mosaic.path)}


def apply_display_windows(layers, windows, dtype) -> None:
    """Give the image layers of a fused result its stored display windows (one per
    channel, in layer order), with the slider over the whole dtype range."""
    images = [layer for layer in layers if hasattr(layer, "contrast_limits")]
    for layer, window in zip(images[-len(windows):] if windows else [], windows):
        if np.issubdtype(np.dtype(dtype), np.integer):
            info = np.iinfo(np.dtype(dtype))
            layer.contrast_limits_range = (info.min, info.max)
        layer.contrast_limits = window


@thread_worker
def _scan_worker(reader, path):
    """Scan the acquisition folder in the background (opens every tile's metadata)."""
    return reader(path)


@thread_worker
def _fuse_worker(tiles, out, kwargs):
    """Run the fusion in the background and return the written dataset."""
    return mos.fuse_tiles(tiles, out, **kwargs)


# ---------------------------------------------------------------------------
# widget
# ---------------------------------------------------------------------------

def mosaic_widget():
    """Create the PyMIF mosaic widget: load raw tiles, show them, fuse them."""
    viewer = current_viewer()
    state = {"mosaic": None, "auto_output": None, "streams": None}

    # ----- console redirection, only while one of our tasks runs -----
    log_widget = QTextEdit()
    log_widget.setReadOnly(True)
    log_widget.setMinimumHeight(110)
    log_widget.setStyleSheet(
        "QTextEdit { background-color: #111; color: #ddd;"
        " font-family: Consolas, Menlo, monospace; font-size: 12px; }"
    )
    stream = EmittingStream()
    stream.text_written.connect(make_log_appender(log_widget))

    def _capture():
        if state["streams"] is None:
            state["streams"] = (sys.stdout, sys.stderr)
            sys.stdout = sys.stderr = stream

    def _release():
        if state["streams"] is not None:
            sys.stdout, sys.stderr = state["streams"]
            state["streams"] = None

    def _busy(busy: bool):
        load_widget.enabled = not busy
        has_mosaic = state["mosaic"] is not None
        show_widget.enabled = has_mosaic and not busy
        fuse_widget.enabled = has_mosaic and not busy
        batch_widget.enabled = has_mosaic and not busy
        ui["fuse_holder"].setEnabled(has_mosaic and not busy)

    # ----- tile map: click a rectangle to pick or unpick that stack position -----
    def _tile_map():
        if viewer is None or state["mosaic"] is None:
            return None
        return find_tile_map(viewer.layers, state["mosaic"].path)

    def _picked() -> Optional[list]:
        return picked_stack_ids(_tile_map())

    def _clear_layers():
        """Remove every layer except the tile map."""
        keep = _tile_map()
        for layer in [layer for layer in viewer.layers if layer is not keep]:
            viewer.layers.remove(layer)

    def _on_map_mouse(_viewer, event):
        """Click: toggle a tile. Shift (pick) or Ctrl (unpick) + drag: every tile touched by
        the box, drawn while dragging.

        A plain drag pans and Alt + drag zooms, as usual in napari.
        """
        layer = _tile_map()
        active = viewer.layers.selection.active
        modifiers = list(event.modifiers)  # vispy keys, equal to their names
        target = (None if not modifiers else True if modifiers == ["Shift"]
                  else False if modifiers == ["Control"] else "other")  # e.g. Alt + drag zooms
        if (layer is None or not layer.visible or viewer.dims.ndisplay != 2 or event.button != 1
                or target == "other"
                or str(getattr(active, "mode", "pan_zoom")) != "pan_zoom"):  # e.g. painting labels
            return
        press, start = np.asarray(event.pos), np.asarray(event.position)
        moved = False
        yield
        while event.type == "mouse_move":
            moved = moved or np.linalg.norm(np.asarray(event.pos) - press) > _CLICK_TOLERANCE
            if moved and target is not None:  # draw the box and preview the picks
                show_drag_box(viewer, layer, start, event.position)
                refresh_tile_map(layer, boxed_picks(layer, tiles_in_box(layer, start, event.position), target))
            yield
        if target is None:  # plain click toggles; a plain drag was a pan
            if not moved and toggle_tile_at(layer, start) is not None:
                _update_plan()
            return
        show_drag_box(viewer, layer)
        if moved:
            indices = tiles_in_box(layer, start, event.position)
        else:
            index = tile_at(layer, start)
            indices = [] if index is None else [index]
        set_picked(layer, indices, target)
        _update_plan()

    if viewer is not None:
        viewer.mouse_drag_callbacks.append(_on_map_mouse)

    # ----- 1. loading -----
    @magicgui(
        call_button="Load tiles",
        microscope={"choices": list(MOSAIC_READERS)},
        input_path={"widget_type": "FileEdit", "mode": "d", "label": "Acquisition folder"},
    )
    def load_widget(microscope="luxendo", input_path: Path = Path()):
        if not input_path or not Path(input_path).is_dir():
            print("Select the acquisition folder first.")
            return
        _capture()
        print(f"Scanning {input_path} ...")
        worker = _scan_worker(MOSAIC_READERS[microscope], input_path)
        _busy(True)

        @worker.returned.connect
        def _done(mosaic):
            state["mosaic"] = mosaic
            _populate(mosaic, Path(input_path))
            _busy(False)
            _release()

        @worker.errored.connect
        def _failed(err):
            print(f"Loading failed: {err}")
            _busy(False)
            _release()

        worker.start()

    def _populate(mosaic, input_path: Path):
        channels, groups = mosaic.channels, group_labels(mosaic)
        for w in (show_widget, fuse_widget):
            w.channels.choices = channels
            w.groups.choices = groups
            w.groups.value = groups
        show_widget.channels.value = channels[:1]  # one channel keeps napari responsive
        fuse_widget.channels.value = channels
        fuse_widget.blend.value = "midline" if len(groups) > 1 and NO_GROUP not in groups else "max"
        lo, hi = mosaic.bounding_box()
        print(f"{len(mosaic.tiles)} tiles, channels {channels}, groups {groups}")
        print("Extent (z, y, x): " + ", ".join(f"{a:.0f}..{b:.0f}" for a, b in zip(lo, hi))
              + f" {mosaic.tiles[0].units}")
        state["input_path"] = input_path
        state["microscope"] = load_widget.microscope.value
        if viewer is not None:
            tile_map = add_tile_map(viewer, mosaic)
            viewer.reset_view()
            print(f"Tile map: {len(tile_map.data)} stack positions, all picked. In the 2D view, click a tile "
                  "to pick or unpick it for showing and fusing; Shift + drag picks, Ctrl + drag unpicks "
                  "every tile touched by a box.")
        batch_widget.batch_csv_path.value = input_path.parent / "mosaic_batch.csv"
        _update_output(force=True)
        _update_plan()

    # ----- 2. preview -----
    @magicgui(
        call_button="Show tiles in napari",
        channels={"widget_type": "Select", "choices": [], "allow_multiple": True},
        groups={"widget_type": "Select", "choices": [], "allow_multiple": True},
        clear_layers={"label": "Remove existing layers", "widget_type": "CheckBox", "value": True,
                      "tooltip": "Remove the layers shown before (the tile map stays)."},
    )
    def show_widget(channels=(), groups=(), clear_layers=True):
        mosaic = state["mosaic"]
        tiles = picked_tiles(mosaic, select_tiles(mosaic, channels, groups), _picked())
        if not tiles:
            print("No picked tile matches the selected channels and groups.")
            return
        _capture()
        try:
            limits = shown_contrast_limits(viewer.layers, mosaic)  # keep the contrast set by hand
            if clear_layers:
                _clear_layers()
            mosaic.visualize(tiles=tiles, viewer=viewer, contrast_limits=limits)
            for layer in tile_layers(viewer.layers, mosaic.path):  # hiding a tile updates the size estimate
                layer.events.visible.connect(_update_plan)
            tile_map = _tile_map()
            if tile_map is not None:  # on top, so its outlines stay visible and clickable
                tile_map.visible = True
                viewer.layers.move(viewer.layers.index(tile_map), len(viewer.layers))
            viewer.reset_view()
            new = [c for c in dict.fromkeys(t.channel for t in tiles) if c not in limits]
            hint = f"; {', '.join(new)} start(s) at the full data range: set the contrast of one tile" if new else ""
            print(f"Showing {len(tiles)} tiles as separate layers (contrast linked per channel{hint}).")
        except Exception as e:
            print(f"Showing tiles failed: {e}")
        finally:
            _release()

    # ----- 3. fusion -----
    @magicgui(
        call_button="Fuse to OME-Zarr",
        channels={"widget_type": "Select", "choices": [], "allow_multiple": True},
        groups={"widget_type": "Select", "choices": [], "allow_multiple": True},
        tiles_to_fuse={"label": "Tiles to fuse", "choices": list(TILE_MODES), "value": "all tiles",
                       "tooltip": "Tiles whose layer is hidden (or not selected) are not read and stay blank; "
                                  "the output keeps its full size. Tiles that are not shown are fused."},
        skip_all_channels={"label": "Blank in all channels", "widget_type": "CheckBox", "value": False,
                           "tooltip": "Leave a skipped tile's position blank in every channel."},
        projection={"choices": list(_PROJECTIONS), "value": "none (3D)"},
        blend={"choices": list(mos.fusion.BLENDS), "value": "max"},
        ds_z={"label": "Downsample Z", "min": 1, "max": 256, "value": 4},
        ds_y={"label": "Downsample Y", "min": 1, "max": 256, "value": 8},
        ds_x={"label": "Downsample X", "min": 1, "max": 256, "value": 8},
        levels={"label": "Resolution levels", "min": 1, "max": 10, "value": 3},
        workers={"label": "Workers", "min": 1, "max": 64, "value": 4},
        overwrite={"label": "Overwrite output", "widget_type": "CheckBox", "value": False},
        open_result={"label": "Open result when done", "widget_type": "CheckBox", "value": True},
        output_path={"widget_type": "FileEdit", "mode": "d", "label": "Output .ome.zarr"},
    )
    def fuse_widget(
        channels=(),
        groups=(),
        tiles_to_fuse="all tiles",
        skip_all_channels=False,
        projection="none (3D)",
        blend="max",
        ds_z=4,
        ds_y=8,
        ds_x=8,
        levels=3,
        workers=4,
        overwrite=False,
        open_result=True,
        output_path: Path = Path(),
    ):
        mosaic = state["mosaic"]
        tiles = picked_tiles(mosaic, select_tiles(mosaic, channels, groups), _picked())
        kwargs = fuse_kwargs(channels=channels, projection=projection, blend=blend, ds_z=ds_z,
                             ds_y=ds_y, ds_x=ds_x, levels=levels,
                             workers=workers, overwrite=overwrite)
        skip = _skipped_tiles()
        kwargs["skip"] = skip
        try:
            plan = mos.plan_fusion(tiles, **{k: kwargs[k] for k in ("channels", "projection", "blend", "ds", "skip")})
        except ValueError as e:
            print(f"Cannot fuse: {e}")
            return
        out = Path(output_path)
        if out.exists() and not overwrite:
            print(f"{out} exists: tick 'Overwrite output' or choose another path.")
            return

        _capture()
        print(f"Fusing {len(plan.tiles)} tiles into {out}: {plan.summary()}")
        worker = _fuse_worker(plan.tiles, out, kwargs)
        _busy(True)

        @worker.returned.connect
        def _done(zm):
            print(f"Fusion completed: {out}")
            _busy(False)
            _release()
            if open_result:
                _clear_layers()
                tile_map = _tile_map()
                if tile_map is not None:  # the result starts at 0, not at the stage position
                    tile_map.visible = False
                zm.visualize(viewer=viewer)
                apply_display_windows(viewer.layers, mos.display_windows(out), zm.metadata["dtype"])

        @worker.errored.connect
        def _failed(err):
            print(f"Fusion failed: {err}")
            _busy(False)
            _release()

        worker.start()

    plan_label = QLabel("Load a mosaic to see the output size.")
    plan_label.setWordWrap(True)
    plan_label.setStyleSheet("QLabel { color: #aaaaaa; }")

    def _current_plan():
        mosaic = state["mosaic"]
        if mosaic is None:
            return None
        v = {name: getattr(fuse_widget, name).value for name in
             ("channels", "projection", "blend", "ds_z", "ds_y", "ds_x")}
        tiles = picked_tiles(mosaic, select_tiles(mosaic, v["channels"], fuse_widget.groups.value), _picked())
        return mos.plan_fusion(
            tiles,
            ds=(v["ds_z"], v["ds_y"], v["ds_x"]),
            projection=_PROJECTIONS[v["projection"]],
            blend=v["blend"],
            channels=list(v["channels"]),
            skip=_skipped_tiles(),
        )

    def _skip_ids() -> list:
        mode = TILE_MODES[fuse_widget.tiles_to_fuse.value]
        if viewer is None or mode is None or state["mosaic"] is None:
            return []
        return skipped_tile_ids(viewer.layers, viewer.layers.selection, state["mosaic"].path, mode,
                                fuse_widget.skip_all_channels.value)

    def _skipped_tiles() -> list:
        ids = _skip_ids()
        return state["mosaic"].tiles_by_ids(ids) if ids else []

    def _update_plan(*_):
        try:
            plan = _current_plan()
        except ValueError as e:
            plan_label.setText(f"Output: {e}")
            return
        if plan is not None:
            plan_label.setText(f"Output: {plan.summary()}")

    def _update_output(*_, force=False):
        if "input_path" not in state:
            return
        new = default_output(state["input_path"], _PROJECTIONS[fuse_widget.projection.value],
                             (fuse_widget.ds_z.value, fuse_widget.ds_y.value, fuse_widget.ds_x.value))
        # Follow the parameters until the user types a path of their own.
        if force or Path(fuse_widget.output_path.value) == state["auto_output"]:
            fuse_widget.output_path.value = new
            state["auto_output"] = new

    for name in ("channels", "groups", "tiles_to_fuse", "skip_all_channels", "projection", "blend",
                 "ds_z", "ds_y", "ds_x"):
        getattr(fuse_widget, name).changed.connect(_update_plan)
    if viewer is not None:
        viewer.layers.selection.events.changed.connect(_update_plan)
        viewer.layers.events.removed.connect(_update_plan)

    # ----- 4. batch CSV export (run later with pymif fuse --batch) -----
    @magicgui(
        call_button="Append to batch CSV",
        batch_csv_path={"widget_type": "FileEdit", "mode": "w", "label": "Batch CSV",
                        "filter": "*.csv"},
    )
    def batch_widget(batch_csv_path: Path = Path("mosaic_batch.csv")):
        mosaic = state["mosaic"]
        values = {name: getattr(fuse_widget, name).value for name in
                  ("channels", "groups", "projection", "blend", "ds_z", "ds_y", "ds_x",
                   "levels", "workers", "overwrite")}
        skip_ids = _skip_ids()
        row = batch_row(input_path=state["input_path"], microscope=state["microscope"],
                        output=fuse_widget.output_path.value, values=values,
                        all_groups=group_labels(mosaic), skip_ids=skip_ids, picked=_picked(),
                        all_stacks={stack_id(mosaic, t) for t in mosaic.tiles})
        _capture()  # print into this widget's log, not wherever sys.stdout points
        try:
            path = append_batch_csv(batch_csv_path, row, MOSAIC_BATCH_COLUMNS)
            only = f", {len(row['tiles'].split())} picked stack position(s)" if row["tiles"] else ""
            blank = f", {len(skip_ids)} tile id(s) left blank" if skip_ids else ""
            print(f"Appended {Path(row['output']).name} to {path}{only}{blank}. Run it with:\n"
                  f"  pymif fuse --batch \"{path}\"")
        except OSError as e:
            print(f"Could not write the batch CSV: {e}")
        finally:
            _release()
    for name in ("projection", "ds_z", "ds_y", "ds_x"):
        getattr(fuse_widget, name).changed.connect(_update_output)

    enable_drag_and_drop(load_widget.input_path, folder_of)
    enable_drag_and_drop(fuse_widget.output_path,
                         lambda p: output_for_drop(p, fuse_widget.output_path.value))

    # ----- layout -----
    ui = {}
    for w in (show_widget, fuse_widget):
        w.channels.native.setMaximumHeight(64)
        w.groups.native.setMaximumHeight(48)
        w.native.layout().setSpacing(2)
        w.native.layout().setContentsMargins(0, 0, 0, 0)

    fuse_button = fuse_widget._call_button.native
    fuse_widget.native.layout().removeWidget(fuse_button)
    ui["fuse_holder"] = make_button_holder(fuse_button)

    def title(text):
        label = QLabel(text)
        label.setStyleSheet(_TITLE_STYLE)
        return label

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setSpacing(6)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.addWidget(make_section(title("Mosaic loading:"), load_widget.native))
    layout.addWidget(make_section(title("Show tiles:"), show_widget.native))
    layout.addWidget(make_section(title("Fuse to OME-Zarr:"), fuse_widget.native, plan_label, ui["fuse_holder"]))
    batch_widget.native.layout().setContentsMargins(0, 0, 0, 0)
    layout.addWidget(make_section(title("Batch CSV export:"), batch_widget.native))
    layout.addStretch(1)

    clear_btn = QPushButton("Clear")
    clear_btn.setMaximumWidth(60)
    clear_btn.clicked.connect(log_widget.clear)
    header = QWidget()
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.addWidget(title("PyMIF Log:"))
    header_layout.addStretch(1)
    header_layout.addWidget(clear_btn)
    log_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    _busy(False)
    dock = compose_dock(container, bottom=make_section(header, log_widget))
    # exposed for tests
    dock.load_widget, dock.show_widget, dock.fuse_widget = load_widget, show_widget, fuse_widget
    dock.batch_widget = batch_widget
    dock.plan_label, dock.state, dock.populate = plan_label, state, _populate
    return dock
