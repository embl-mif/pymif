# tests/test_manager_refactor.py
"""Tests for the shared manager helpers and the vendor readers (synthetic data)."""
from __future__ import annotations

import warnings

import dask.array as da
import h5py
import numpy as np
import pytest
import tifffile

import pymif.microscope_manager as mm
from pymif.utils.axes import to_tczyx
from pymif.utils.dataset_ops import (
    apply_metadata_updates,
    infer_downscale_factors,
    known_updates,
    reorder_channels,
    subset_levels,
)
from pymif.utils.metadata import ChannelInfo, build_metadata, scale_for_level
from pymif.utils.ngff import parse_channel_color
from pymif.utils.ome import parse_ome_xml
from pymif.utils.units import normalize_unit, to_micrometers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, index, expected",
    [
        (None, 0, "FFFFFF"),
        ("", 1, "FF0000"),
        (None, 7, "FFFFFF"),  # palette cycles
        (255, 0, "0000FF"),  # 24-bit RGB (Opera)
        ("16711680", 0, "FF0000"),
        (-1, 0, "FFFFFF"),  # signed RGBA
        (-16776961, 0, "FF0000"),
        ("#80FF0000", 0, "FF0000"),  # Zeiss #AARRGGBB
        ("#00ff00", 0, "00FF00"),
        ("magenta", 0, "FF00FF"),
    ],
)
def test_parse_channel_color(raw, index, expected):
    assert parse_channel_color(raw, index) == expected


def test_parse_channel_color_invalid():
    with pytest.raises(TypeError):
        parse_channel_color("notacolor")


def test_units():
    assert normalize_unit("µm") == "micrometer"
    assert normalize_unit("um") == "micrometer"
    assert normalize_unit("s") == "second"
    assert normalize_unit(None) is None
    assert normalize_unit("parsec") == "parsec"
    scales, units = to_micrometers((1e-6, 2.0, 0.5), ("m", "mm", "µm"))
    assert scales == pytest.approx((1.0, 2000.0, 0.5))
    assert units == ("micrometer",) * 3
    assert to_micrometers((3.0,), ("pixel",)) == ((3.0,), ("pixel",))


def test_to_tczyx_inserts_and_orders():
    arr = da.zeros((4, 3, 5, 6), chunks=2)  # c z y x -> shuffled below
    out = to_tczyx(arr, "zcyx")
    assert out.shape == (1, 3, 4, 5, 6)
    assert to_tczyx(da.zeros((5, 6)), "YX").shape == (1, 1, 1, 5, 6)
    with pytest.raises(ValueError):
        to_tczyx(da.zeros((2, 2)), "qx")


def test_scale_for_level():
    assert scale_for_level((2.0, 0.5, 0.5), (8, 32, 32), (4, 16, 16)) == (4.0, 1.0, 1.0)


def test_build_metadata_normalizes():
    md = build_metadata(
        size=[(1, 2, 3, 4, 5)],
        scales=[(1, 1, 1)],
        units=("µm",) * 3,
        channels=[ChannelInfo("a", "#00ff00"), ChannelInfo("b")],
        dtype=np.dtype("uint8"),
        time_increment_unit="s",
        extra_key=1,
    )
    assert md["units"] == ("micrometer",) * 3
    assert md["time_increment_unit"] == "second"
    assert md["channel_colors"] == ["00FF00", "FF0000"]
    assert md["dtype"] == "uint8" and md["axes"] == "tczyx" and md["extra_key"] == 1


OME = """<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image><Pixels
 SizeT="2" SizeC="2" SizeZ="3" SizeY="8" SizeX="9" PhysicalSizeX="0.5" PhysicalSizeY="0.5"
 PhysicalSizeZ="2" PhysicalSizeXUnit="µm" PhysicalSizeYUnit="µm" PhysicalSizeZUnit="µm" Type="uint16">
 <Channel ID="C:0" Name="A" Color="16711935"/><Channel ID="C:1"/></Pixels></Image></OME>"""


def test_parse_ome_xml_colors_and_defaults():
    px = parse_ome_xml(OME)  # spec-compliant RGBA: 0x00FF00FF is green
    assert (px.size_t, px.size_c, px.size_z, px.size_y, px.size_x) == (2, 2, 3, 8, 9)
    assert px.scale_zyx == (2.0, 0.5, 0.5)
    assert px.channels[0].color == "00FF00"
    assert px.channels[1].name == "Channel 1" and px.channels[1].color == "FFFFFF"
    # Opera-style 24-bit RGB: 0x00FF00FF is magenta
    assert parse_channel_color(parse_ome_xml(OME, color_format="auto").channels[0].color) == "FF00FF"


def test_parse_ome_xml_errors_and_warnings():
    with pytest.raises(ValueError, match="SizeY"):
        parse_ome_xml('<OME xmlns="x"><Pixels SizeX="2"/></OME>')
    with pytest.warns(UserWarning, match="PhysicalSizeZ"):
        px = parse_ome_xml('<Pixels SizeX="2" SizeY="2" PhysicalSizeX="1" PhysicalSizeY="1"/>')
    assert px.scale_zyx[0] == 1.0


# ---------------------------------------------------------------------------
# dataset_ops
# ---------------------------------------------------------------------------

def test_dataset_ops_roundtrip(image_pyramid, metadata):
    md = dict(metadata)
    assert infer_downscale_factors(md) == 2

    data = reorder_channels(image_pyramid, md, [1, 0])
    assert md["channel_names"] == ["B", "A"]
    assert np.array_equal(data[0][:, 0].compute(), image_pyramid[0][:, 1].compute())
    with pytest.raises(ValueError):
        reorder_channels(image_pyramid, md, [0, 0])

    new_data, new_md = subset_levels(data, md, T=[0], Z=slice(0, 2))
    assert len(new_data) == 3  # pyramid rebuilt with the same number of levels
    assert new_md["size"][0][:3] == (1, 2, 2)
    with pytest.raises(ValueError, match="out of range"):
        subset_levels(data, md, C=[5])


def test_apply_metadata_updates_validation(image_pyramid, metadata):
    md = dict(metadata)
    apply_metadata_updates(image_pyramid, md, known_updates({"channel_colors": ["magenta", "cyan"], "time_increment": 2}))
    assert md["channel_colors"] == ["FF00FF", "00FFFF"] and md["time_increment"] == 2

    with pytest.warns(UserWarning, match="unknown"):
        assert known_updates({"bogus": 1}) == {}
    with pytest.warns(UserWarning, match="does not match"):
        apply_metadata_updates(image_pyramid, md, {"channel_names": ["only one"]})
    with pytest.raises(ValueError):
        apply_metadata_updates(image_pyramid, md, {"time_increment": -1})
    with pytest.raises(ValueError):
        apply_metadata_updates(image_pyramid, md, {"units": ("a",)})


# ---------------------------------------------------------------------------
# Base-class contract
# ---------------------------------------------------------------------------

def test_array_manager_read_contract(image_pyramid, metadata):
    m = mm.ArrayManager(image_pyramid, metadata)
    assert m.read() is None
    assert len(m.data) == 3 and m.metadata["axes"] == "tczyx"
    assert m.chunks == (1, 1, 8, 4096, 4096)
    assert mm.ArrayManager(image_pyramid[0], dict(metadata, scales=[(1, 1, 1)])).metadata["scales"] == [(1, 1, 1)]


def test_context_manager_closes():
    class Handle:
        closed = False

        def close(self):
            self.closed = True

    class Dummy(mm.MicroscopeManager):
        def _parse_metadata(self):
            return {}

        def _build_dask_array(self):
            return []

    h = Handle()
    with Dummy() as d:
        d._open_files.append(h)
    assert h.closed and d._open_files == []
    assert Dummy().chunks == mm.MicroscopeManager.DEFAULT_CHUNKS


def test_base_manager_cannot_be_instantiated():
    with pytest.raises(TypeError):
        mm.MicroscopeManager()


# ---------------------------------------------------------------------------
# Vendor readers on synthetic datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def luxendo_dir(tmp_path):
    xml = """<SpimData><SequenceDescription>
    <ViewSetups><ViewSetup><id>0</id><size>32 32 8</size><voxelSize><unit>micrometer</unit><size>0.5 0.5 2</size></voxelSize></ViewSetup>
    <Attributes name="channel"><Channel><id>0</id><name>ch:0</name></Channel><Channel><id>1</id><name>ch:1</name></Channel></Attributes>
    </ViewSetups><Timepoints type="range"><first>0</first><last>1</last></Timepoints></SequenceDescription></SpimData>"""
    (tmp_path / "main.xml").write_text(xml)
    rng = np.random.default_rng(0)
    for t in range(2):
        for c in range(2):
            with h5py.File(tmp_path / f"uni_tp-{t}_ch-{c}_st-0.lux.h5", "w") as f:
                f["Data"] = rng.integers(0, 100, (8, 32, 32), dtype=np.uint16)
                f["Data222"] = rng.integers(0, 100, (4, 16, 16), dtype=np.uint16)
    return tmp_path


def test_luxendo_manager(luxendo_dir):
    with mm.LuxendoManager(luxendo_dir, chunks=(1, 1, 4, 16, 16)) as lux:
        assert [a.shape for a in lux.data] == [(2, 2, 8, 32, 32), (2, 2, 4, 16, 16)]
        assert lux.metadata["scales"] == [(2.0, 0.5, 0.5), (4.0, 1.0, 1.0)]  # from shapes, not names
        assert lux.metadata["channel_names"] == ["ch:0", "ch:1"]
        assert lux.metadata["channel_colors"] == ["FFFFFF", "FF0000"]
        assert lux.metadata["dtype"] == "uint16" and lux.metadata["units"] == ("micrometer",) * 3
        with h5py.File(luxendo_dir / "uni_tp-1_ch-0_st-0.lux.h5") as f:
            assert np.array_equal(lux.data[0][1, 0].compute(), f["Data"][()])
    assert lux._open_files == []


def test_luxendo_missing_file_is_reported(luxendo_dir):
    (luxendo_dir / "uni_tp-1_ch-1_st-0.lux.h5").unlink()
    with pytest.raises(ValueError, match="missing"):
        mm.LuxendoManager(luxendo_dir)
    with pytest.raises(FileNotFoundError):
        mm.LuxendoManager(luxendo_dir / "nowhere")


def test_luxendo_non_contiguous_ids_are_renumbered(luxendo_dir):
    """File names carry acquisition IDs (e.g. a fused 647 channel saved as ``ch-4``), not indices."""
    for f in list(luxendo_dir.glob("*.lux.h5")):
        t, c = mm.LuxendoManager._tp_ch(f.name)
        f.rename(luxendo_dir / f"uni_tp-{t + 3}_ch-{[1, 4][c]}_st-0.lux.h5")
    with mm.LuxendoManager(luxendo_dir) as lux:
        assert lux.data[0].shape == (2, 2, 8, 32, 32)
        with h5py.File(luxendo_dir / "uni_tp-4_ch-4_st-0.lux.h5") as f:
            assert np.array_equal(lux.data[0][1, 1].compute(), f["Data"][()])


@pytest.fixture
def viventis_dir(tmp_path):
    tiffdata = "".join(
        f'<TiffData FirstT="{t}" FirstC="{c}"><UUID FileName="t{t}_c{c}.tif"/></TiffData>'
        for t in range(2) for c in range(2)
    )
    (tmp_path / "x.companion.ome").write_text(OME.replace("</Pixels>", tiffdata + "</Pixels>"), encoding="utf-8")
    for t in range(2):
        for c in range(2):
            tifffile.imwrite(tmp_path / f"t{t}_c{c}.tif", np.full((3, 8, 9), 10 * t + c, np.uint16))
    return tmp_path


def test_viventis_manager(viventis_dir):
    v = mm.ViventisManager(viventis_dir)
    assert v.data[0].shape == (2, 2, 3, 8, 9)
    assert v.metadata["scales"] == [(2.0, 0.5, 0.5)]
    assert v.metadata["channel_colors"] == ["00FF00", "FFFFFF"]
    assert v.metadata["plane_files"][(1, 1)] == "t1_c1.tif"
    assert int(v.data[0][1, 1, 0, 0, 0].compute()) == 11


def test_viventis_errors(viventis_dir):
    (viventis_dir / "x.companion.ome").unlink()
    with pytest.raises(FileNotFoundError):
        mm.ViventisManager(viventis_dir)


def test_opera_manager(tmp_path):
    path = tmp_path / "opera.ome.tiff"
    data = np.arange(2 * 3 * 4 * 8 * 8, dtype=np.uint16).reshape(2, 3, 4, 8, 8)
    tifffile.imwrite(
        path, data, ome=True,
        metadata={"axes": "TCZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "PhysicalSizeZ": 3.0,
                  "Channel": {"Name": ["a", "b", "c"], "Color": [255, 16711680, 65280]}},
    )
    o = mm.OperaManager(path, chunks=(1, 1, 2, 8, 8))
    assert o.data[0].shape == (2, 3, 4, 8, 8)
    assert np.array_equal(o.data[0].compute(), data)
    assert o.metadata["scales"][0] == pytest.approx((3.0, 0.5, 0.5))
    assert o.metadata["channel_names"] == ["a", "b", "c"]
    assert len(o.metadata["channel_colors"]) == 3
    assert o.metadata["units"] == ("micrometer",) * 3


def test_scape_manager(tmp_path):
    (tmp_path / "Metadata").mkdir()
    dims = "".join(
        f'<DimensionDescription DimID="{i}" NumberOfElements="{n}" Length="{n * 1e-6}" Unit="m"/>'
        for i, n in [(1, 8), (2, 8), (3, 3), (4, 2)]
    )
    (tmp_path / "Metadata" / "p1 (2).xlif").write_text(
        f'<Root><ImageDescription><Dimensions>{dims}</Dimensions><Channels>'
        '<ChannelDescription LUTName="Green"/></Channels></ImageDescription></Root>'
    )
    path = tmp_path / "p1 (2).ome.tif"
    data = np.arange(2 * 3 * 8 * 8, dtype=np.uint16).reshape(2, 3, 8, 8)
    tifffile.imwrite(path, data, metadata={"axes": "TZYX"})
    s = mm.ScapeManager(path, chunks=(1, 1, 3, 8, 8))
    assert s.data[0].shape == (2, 1, 3, 8, 8)
    assert np.array_equal(s.data[0][:, 0].compute(), data)
    assert s.metadata["scales"][0] == pytest.approx((1.0, 1.0, 1.0))
    assert s.metadata["units"] == ("micrometer",) * 3
    assert s.metadata["channel_colors"] == ["00FF00"]
    assert mm.ScapeManager._strip_ome_tiff_suffix("a.b.OME.TIFF") == "a.b"
    assert mm.ScapeManager._strip_ome_tiff_suffix("a.png") == "a"


def test_vendor_manager_roundtrip_to_zarr(tmp_path, viventis_dir):
    out = tmp_path / "out.zarr"
    v = mm.ViventisManager(viventis_dir)
    v.update_metadata({"channel_colors": ["red", "blue"]})
    v.subset_dataset(T=[0], rebuild_pyramid=False)
    v.to_zarr(str(out), overwrite=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = mm.ZarrManager(str(out))
    assert z.metadata["channel_colors"] == ["FF0000", "0000FF"]


def test_scape_accepts_path_and_legacy_keyword(tmp_path):
    (tmp_path / "Metadata").mkdir()
    dims = "".join(
        f'<DimensionDescription DimID="{i}" NumberOfElements="{n}" Length="{n * 1e-6}" Unit="m"/>'
        for i, n in [(1, 4), (2, 4), (3, 1), (4, 1)]
    )
    (tmp_path / "Metadata" / "a.xlif").write_text(
        f"<Root><ImageDescription><Dimensions>{dims}</Dimensions></ImageDescription></Root>"
    )
    tifffile.imwrite(tmp_path / "a.ome.tif", np.zeros((4, 4), np.uint16))
    assert mm.ScapeManager(path=tmp_path / "a.ome.tif").data[0].shape == (1, 1, 1, 4, 4)
    assert mm.ScapeManager(ome_tiff_path=tmp_path / "a.ome.tif").data[0].shape == (1, 1, 1, 4, 4)
    with pytest.raises(TypeError):
        mm.ScapeManager()


# ---------------------------------------------------------------------------
# Shared utils (colors, chunks, downsampling, array creation) and progress bar
# ---------------------------------------------------------------------------

def test_color_helpers_single_source():
    from pymif.utils import colors, ngff
    from pymif.utils.visualize import _parse_color

    # historical import locations keep working and point to the same objects
    assert ngff.parse_color is colors.parse_color
    assert ngff.ome_color_to_hex is colors.ome_color_to_hex
    assert colors.hex_to_rgb("FF8000") == (1.0, 128 / 255, 0.0)
    assert _parse_color("0xFF0000") == (1.0, 0.0, 0.0)
    assert _parse_color(0x00FF00) == (0.0, 1.0, 0.0)
    assert _parse_color("#80FF00FF") == (1.0, 0.0, 1.0)  # AARRGGBB, alpha dropped
    assert _parse_color("cyan") == (0.0, 1.0, 1.0)


def test_downsample_nearest_numpy_and_dask_agree():
    from pymif.utils.downsampling import downsample_nearest

    arr = np.arange(2 * 5 * 7).reshape(2, 5, 7)
    out_np = downsample_nearest(arr, (2, 2), spatial_axes=(1, 2))
    out_da = downsample_nearest(da.from_array(arr, chunks=3), (2, 2), spatial_axes=(1, 2))
    assert out_np.shape == (2, 3, 4)  # ceil(n / f) with edge padding
    assert isinstance(out_da, da.Array)
    assert np.array_equal(out_np, out_da.compute())
    assert np.array_equal(downsample_nearest(arr, (1, 1), (1, 2)), arr)  # factor 1: untouched


def test_chunk_helpers():
    from pymif.utils.chunks import get_chunks, rechunk_to_shape, shape_tuple

    assert shape_tuple((1, 2), 2) == (1, 2) and shape_tuple((1, 2), 3) is None
    assert shape_tuple("auto", 2) is None and shape_tuple((0, 2), 2) is None
    arr = da.zeros((10, 10), chunks=(5, 5))
    assert rechunk_to_shape(arr, (5, 5)) is arr
    assert get_chunks(rechunk_to_shape(arr, (100, 4))) == (10, 4)


def _image_md(shards):
    return {
        "size": [(1, 2, 8, 32, 32)], "chunksize": [(1, 1, 4, 16, 16)], "scales": [(1.0, 0.5, 0.5)],
        "units": ("micrometer",) * 3, "axes": "tczyx", "channel_names": ["a", "b"],
        "channel_colors": ["FF0000", "00FF00"], "time_increment": 1.0, "time_increment_unit": "second",
        "dtype": "uint16", "shards": shards,
    }


def test_labels_created_from_image_metadata_with_shards(tmp_path):
    """Image metadata read from a sharded store can seed a channel-less label group."""
    root = mm.ZarrManager(str(tmp_path / "s.zarr"), mode="a", metadata=_image_md(None))
    md = dict(root.metadata)
    md["shards"] = [(1, 1, 8, 32, 32)]  # 5-D shards inherited from the image

    # legacy path (is_label=True drops the channel axis) keeps shards consistent
    legacy_md = {k: v for k, v in md.items() if k != "data_type"}  # old callers passed no data_type
    root.create_empty_group("legacy", legacy_md, is_label=True)
    assert root.root["labels/legacy/0"].shards == (1, 8, 32, 32)

    # explicit 4-D label metadata with stale 5-D shards: ignored with a warning
    label_md = dict(md, axes="tzyx", size=[(1, 8, 32, 32)], chunksize=[(1, 4, 16, 16)],
                    channel_names=[], channel_colors=[], data_type="label")
    with pytest.warns(UserWarning, match="Ignoring metadata"):
        root.create_empty_group("explicit", label_md, is_label=True)
    assert root.root["labels/explicit/0"].shape == (1, 8, 32, 32)


def test_progress_bar_toggle(tmp_path, image_pyramid, metadata, capsys):
    mm.ArrayManager(image_pyramid, metadata).to_zarr(str(tmp_path / "bar.zarr"), progress=True)
    shown = capsys.readouterr().err
    assert "bar.zarr" in shown and "100%" in shown

    mm.ArrayManager(image_pyramid, metadata).to_zarr(str(tmp_path / "quiet.zarr"), progress=False)
    assert capsys.readouterr().err == ""

    # compute=False must not compute (and so must not show a bar)
    mm.ArrayManager(image_pyramid, metadata).to_zarr(str(tmp_path / "lazy.zarr"), compute=False)
    assert capsys.readouterr().err == ""


def test_progress_bar_labels_each_dataset(tmp_path, image_pyramid, metadata, label_pyramid, capsys):
    z = mm.ZarrManager(str(tmp_path / "multi.zarr"), mode="a", metadata=metadata)
    z.write_image_region(np.zeros((2, 2, 4, 16, 16), np.uint16), level=0)
    capsys.readouterr()
    z.to_zarr(str(tmp_path / "copy.zarr"), zarr_format=2, ngff_version="0.4", overwrite=True)
    assert "raw" in capsys.readouterr().err


def test_visualize_layers_carry_units_matching_converted_zarr(tmp_path, image_pyramid, metadata):
    """Source layers and the re-imported zarr must agree on units.

    Otherwise napari warns "Inconsistent units across layers" and stops using
    units for rendering (e.g. the scale bar).
    """
    pytest.importorskip("napari")
    ome_zarr = pytest.importorskip("napari_ome_zarr")
    from napari.components import ViewerModel

    source = mm.ArrayManager(image_pyramid, dict(metadata, units=("µm",) * 3, time_increment_unit="s"))
    out = tmp_path / "units.zarr"
    source.to_zarr(str(out), progress=False)

    viewer = ViewerModel()
    source.visualize(viewer=viewer)
    source_units = viewer.layers.extent.units
    assert source_units is not None
    assert [str(u) for u in source_units] == ["second", "micrometer", "micrometer", "micrometer"]

    for data, kwargs, kind in ome_zarr.napari_get_reader(str(out))(str(out)):
        getattr(viewer, "add_" + kind)(data, **kwargs)
    assert viewer.layers.extent.units == source_units  # consistent -> no warning


def test_luxendo_pyramid_levels_with_resampled_z(luxendo_dir):
    """Luxendo levels can have *more* z planes than level 0; scales must keep the physical extent."""
    for f in luxendo_dir.glob("*.lux.h5"):
        with h5py.File(f, "a") as h:
            del h["Data222"]
            h["Data_2_2_1"] = np.zeros((12, 16, 16), np.uint16)  # z: 8 -> 12, xy halved
    with mm.LuxendoManager(luxendo_dir) as lux:
        assert [a.shape[2:] for a in lux.data] == [(8, 32, 32), (12, 16, 16)]
        assert lux.metadata["scales"][1] == pytest.approx((2.0 * 8 / 12, 1.0, 1.0))
        extents = [tuple(n * s for n, s in zip(sz[2:], sc)) for sz, sc in zip(lux.metadata["size"], lux.metadata["scales"])]
        assert extents[0] == pytest.approx(extents[1])


def test_plugin_helper_layers_use_dataset_units(image_pyramid, metadata):
    """ROI / Zrange / CropBox layers must share the image layers' units (else napari warns)."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    from pymif.napari._dataset_helpers import axis_size, scale_for_axes, units_kwargs

    dataset = mm.ArrayManager(image_pyramid, dict(metadata, time_increment_unit="s"))
    viewer = ViewerModel()
    dataset.visualize(viewer=viewer)

    ymax, xmax = axis_size(dataset, "y") - 1, axis_size(dataset, "x") - 1
    viewer.add_shapes(
        np.array([[0, 0], [0, xmax], [ymax, xmax], [ymax, 0]]), shape_type="rectangle", name="ROI",
        ndim=2, scale=scale_for_axes(dataset, "yx"), **units_kwargs(dataset, "yx"),
    )
    viewer.add_points(
        np.array([[0, 1, 1], [2, 1, 1]]), name="Zrange", ndim=3,
        scale=scale_for_axes(dataset, "zyx"), **units_kwargs(dataset, "zyx"),
    )
    assert viewer.layers.extent.units is not None  # None would trigger napari's warning

    # the helper builds exactly what napari expects for each axis subset
    assert units_kwargs(dataset, "yx")["units"] == ("micrometer", "micrometer")
    assert units_kwargs(dataset, "tyx")["units"][0] == "second"
    assert units_kwargs(dataset, "q") == {"units": ()}  # axes the dataset lacks are skipped


def test_visualize_contrast_limits_do_not_overflow_uint16(metadata):
    """2 * np.uint16(40000) wraps to 14464; the limits must be computed with Python ints."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    data = np.zeros((2, 2, 4, 16, 16), np.uint16)
    data[..., 0, 0] = 40000
    manager = mm.ArrayManager(data, dict(metadata, size=[data.shape], chunksize=[(1, 1, 2, 8, 8)], scales=[(2.0, 0.5, 0.5)]))

    viewer = ViewerModel()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # "overflow encountered in scalar multiply"
        manager.visualize(viewer=viewer)
    layer = viewer.layers[0]
    assert layer.contrast_limits[1] >= 40000  # wrapped value would have been 14464


def test_points_layer_has_no_deprecation_warning():
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    from pymif.napari._dataset_helpers import points_projection_kwargs

    from napari.layers import Points

    kwargs = points_projection_kwargs()
    if "rescale_linear" in {str(m) for m in getattr(Points, "_projectionclass", ())}:
        assert "out_of_slice_display" not in kwargs  # deprecated since napari 0.9
    else:  # napari < 0.9 (0.7 has projection_mode, but without rescale_linear)
        assert kwargs == {"out_of_slice_display": True}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ViewerModel().add_points(np.array([[0, 1, 1], [2, 1, 1]]), ndim=3, **kwargs)
