"""napari widget to inspect raw mosaic tiles and fuse them into an OME-Zarr image."""

import sys
from pathlib import Path
from typing import Optional, Sequence

from magicgui import magicgui
from napari import current_viewer
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget

from pymif import mosaic_manager as mos
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
    downsample: str,
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
        "downsample": downsample,
        "levels": int(levels),
        "workers": int(workers),
        "overwrite": bool(overwrite),
    }


@thread_worker
def _scan_worker(reader, path):
    """Scan the acquisition folder in the background (opens every tile's metadata)."""
    return reader(path)


@thread_worker
def _contrast_worker(mosaic, tiles):
    """Read the small contrast sample in the background (it is slow on network drives)."""
    return mosaic.contrast_limits(tiles)


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
        ui["fuse_holder"].setEnabled(has_mosaic and not busy)

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
        _update_output(force=True)
        _update_plan()

    # ----- 2. preview -----
    @magicgui(
        call_button="Show tiles in napari",
        channels={"widget_type": "Select", "choices": [], "allow_multiple": True},
        groups={"widget_type": "Select", "choices": [], "allow_multiple": True},
        clear_layers={"label": "Remove existing layers", "widget_type": "CheckBox", "value": True},
    )
    def show_widget(channels=(), groups=(), clear_layers=True):
        mosaic = state["mosaic"]
        tiles = select_tiles(mosaic, channels, groups)
        if not tiles:
            print("No tile matches the selected channels and groups.")
            return
        _capture()
        print(f"Estimating contrast from a few tiles of {', '.join(dict.fromkeys(t.channel for t in tiles))} ...")
        worker = _contrast_worker(mosaic, tiles)
        _busy(True)

        @worker.returned.connect
        def _done(limits):
            for channel, (low, high) in limits.items():
                print(f"{channel}: contrast {low:.0f} - {high:.0f} (linked across its tiles)")
            if clear_layers:
                viewer.layers.clear()
            mosaic.visualize(tiles=tiles, viewer=viewer, contrast_limits=limits)
            viewer.reset_view()
            print(f"Showing {len(tiles)} tiles as separate layers.")
            _busy(False)
            _release()

        @worker.errored.connect
        def _failed(err):
            print(f"Showing tiles failed: {err}")
            _busy(False)
            _release()

        worker.start()

    # ----- 3. fusion -----
    @magicgui(
        call_button="Fuse to OME-Zarr",
        channels={"widget_type": "Select", "choices": [], "allow_multiple": True},
        groups={"widget_type": "Select", "choices": [], "allow_multiple": True},
        projection={"choices": list(_PROJECTIONS), "value": "none (3D)"},
        blend={"choices": list(mos.fusion.BLENDS), "value": "max"},
        ds_z={"label": "Downsample Z", "min": 1, "max": 256, "value": 4},
        ds_y={"label": "Downsample Y", "min": 1, "max": 256, "value": 8},
        ds_x={"label": "Downsample X", "min": 1, "max": 256, "value": 8},
        downsample={"label": "Downsampling", "choices": list(mos.fusion.DOWNSAMPLINGS), "value": "mean"},
        levels={"label": "Resolution levels", "min": 1, "max": 10, "value": 3},
        workers={"label": "Workers", "min": 1, "max": 64, "value": 4},
        overwrite={"label": "Overwrite output", "widget_type": "CheckBox", "value": False},
        open_result={"label": "Open result when done", "widget_type": "CheckBox", "value": True},
        output_path={"widget_type": "FileEdit", "mode": "d", "label": "Output .ome.zarr"},
    )
    def fuse_widget(
        channels=(),
        groups=(),
        projection="none (3D)",
        blend="max",
        ds_z=4,
        ds_y=8,
        ds_x=8,
        downsample="mean",
        levels=3,
        workers=4,
        overwrite=False,
        open_result=True,
        output_path: Path = Path(),
    ):
        mosaic = state["mosaic"]
        tiles = select_tiles(mosaic, channels, groups)
        kwargs = fuse_kwargs(channels=channels, projection=projection, blend=blend, ds_z=ds_z,
                             ds_y=ds_y, ds_x=ds_x, downsample=downsample, levels=levels,
                             workers=workers, overwrite=overwrite)
        try:
            plan = mos.plan_fusion(tiles, **{k: kwargs[k] for k in ("channels", "projection", "blend", "ds")})
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
                viewer.layers.clear()
                zm.visualize(viewer=viewer)

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
        tiles = select_tiles(mosaic, v["channels"], fuse_widget.groups.value)
        return mos.plan_fusion(
            tiles,
            ds=(v["ds_z"], v["ds_y"], v["ds_x"]),
            projection=_PROJECTIONS[v["projection"]],
            blend=v["blend"],
            channels=list(v["channels"]),
        )

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

    for name in ("channels", "groups", "projection", "blend", "ds_z", "ds_y", "ds_x"):
        getattr(fuse_widget, name).changed.connect(_update_plan)
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
    dock.plan_label, dock.state, dock.populate = plan_label, state, _populate
    return dock
