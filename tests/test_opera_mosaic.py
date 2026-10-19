from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile

import pymif.microscope_manager as mm
from pymif.cli import pymif as cli
from pymif.mosaic_manager import OperaFlatfield, OperaMosaicManager, TiffStackArray, fusion
from pymif.mosaic_manager import opera_mosaic_manager as opera

NY, NX = 8, 10  # rows, columns of every plane
PROFILE = ("{Background: {Character: NonFlat, Mean: 100.0, NoiseConst: 1.3, NonFlatness: {Corrected: 0.1, "
           "Original: 0.3}, Profile: {Coefficients: [[1.0], [0.2, -0.1]], Dims: [10, 8], Origin: [4.5, 3.5], "
           "Scale: [0.1, 0.125], Type: Polynomial}, Quality: 1.0}, Channel: 1, ChannelName: EGFP, "
           "Version: Acapella:2013}")
IDENTITY = ("{Background: {Character: Null, Profile: {Type: Identity}, Quality: 1}, Channel: 2, "
            "ChannelName: Brightfield, Version: Acapella:2013}")
MATRIX = "[[0.5,0,0,-17.2],[0,-0.5,0,4.3],[0,0,1.33,3.097]]"  # 1 um pixels calibrated to 0.5 um, y flipped

#: (row, col, field, X m, Y m): fields 1, 2, 4 of well A1 touch; field 3 is far away; well B1 has its own field 1
FIELDS = [(1, 1, 1, 0.0, 0.0), (1, 1, 2, 4e-6, 0.0), (1, 1, 4, 0.0, 3e-6), (1, 1, 3, 1e-3, 0.0), (2, 1, 1, 0.0, 0.0)]
PLANES_Z = (-2e-6, 0.0, 2e-6)
FAILED = (1, 1, 3, 2, 1)  # row, col, field, plane, channel


def raw_plane(field: int, plane: int, channel: int) -> np.ndarray:
    base = np.arange(NY * NX, dtype=np.uint16).reshape(NY, NX) * 3 + 50  # 50..287, around the background
    return base + 1000 * (channel - 1) + 10 * field + plane


def _channel_entries(flatfield: bool) -> str:
    out = []
    for ch, name, profile in ((1, "EGFP", PROFILE), (2, "Brightfield", IDENTITY)):
        if flatfield:
            out.append(f'<Map><Entry ChannelID="{ch}"><FlatfieldProfile>{profile}</FlatfieldProfile></Entry></Map>')
        out.append(
            f'<Map><Entry ChannelID="{ch}"><ChannelName>{name}</ChannelName>'
            '<ImageResolutionX Unit="m">1E-06</ImageResolutionX><ImageResolutionY Unit="m">1E-06</ImageResolutionY>'
            f"<ImageSizeX>{NX}</ImageSizeX><ImageSizeY>{NY}</ImageSizeY>"
            f"<OrientationMatrix>{MATRIX}</OrientationMatrix></Entry></Map>")
    return "\n".join(out)


def write_export(root, flatfield_in_index: bool = True, plate=(8, 12)):
    """A tiny Harmony export: ``root/Images/Index.xml`` and one TIFF per plane."""
    images = root / "Images"
    images.mkdir(parents=True)
    ids, entries = [], []
    for row, col, field, x, y in FIELDS:
        for p, z in enumerate(PLANES_Z, start=1):
            for ch in (1, 2):
                url = f"r{row:02d}c{col:02d}f{field:02d}p{p:02d}-ch{ch}sk1fk1fl1.tiff"
                failed = (row, col, field, p, ch) == FAILED
                if not failed:
                    tifffile.imwrite(images / url, raw_plane(field, p, ch))
                image_id = f"{row:02d}{col:02d}K1F{field}P{p}R{ch}"
                ids.append(f'<Image id="{image_id}" />')
                entries.append(
                    f'<Image Version="1"><id>{image_id}</id><State>{"Failed" if failed else "Ok"}</State>'
                    f"<URL>{url}</URL><Row>{row}</Row><Col>{col}</Col><FieldID>{field}</FieldID>"
                    f"<PlaneID>{p}</PlaneID><TimepointID>1</TimepointID><ChannelID>{ch}</ChannelID>"
                    f'<PositionX Unit="m">{x}</PositionX><PositionY Unit="m">{y}</PositionY>'
                    f'<PositionZ Unit="m">{z}</PositionZ><AbsPositionZ Unit="m">{0.1 + z + field * 1e-5}</AbsPositionZ>'
                    "</Image>")
    plates = (f"<Plates><Plate><PlateID>plate</PlateID><PlateRows>{plate[0]}</PlateRows>"
              f"<PlateColumns>{plate[1]}</PlateColumns></Plate></Plates>\n") if plate else ""
    (images / "Index.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<EvaluationInputData Version="2" xmlns="http://www.perkinelmer.com/PEHH/HarmonyV6">\n'
        f"{plates}<Wells><Well><id>0101</id>{''.join(ids)}</Well></Wells>\n"
        f"<Maps>{_channel_entries(flatfield_in_index)}</Maps>\n"
        f"<Images>{''.join(entries)}</Images>\n</EvaluationInputData>\n",
        encoding="utf-8")
    return root


def expected_flatfield(raw: np.ndarray) -> np.ndarray:
    """Harmony's correction with PROFILE, written out pixel by pixel."""
    out = np.empty(raw.shape)
    for y in range(raw.shape[0]):
        for x in range(raw.shape[1]):
            u, v = (x - 4.5) * 0.1, (y - 3.5) * 0.125
            p = 1.0 + 0.2 * u - 0.1 * v
            r = float(raw[y, x])
            out[y, x] = r - 100.0 * (p - 1) if r >= 100.0 * p else r / p
    return np.rint(out)


@pytest.fixture
def export(tmp_path):
    return write_export(tmp_path / "plate-Measurement 1")


@pytest.fixture(autouse=True)
def small_regions(monkeypatch):
    monkeypatch.setattr(fusion, "_REGION_3D", (4, 4))
    monkeypatch.setattr(fusion, "_REGION_2D", (4, 4))


def test_parse_acapella():
    p = opera.parse_acapella(PROFILE)
    assert p["Version"] == "Acapella:2013" and p["ChannelName"] == "EGFP" and p["Channel"] == 1
    assert p["Background"]["Profile"]["Coefficients"] == [[1.0], [0.2, -0.1]]
    assert p["Background"]["NonFlatness"] == {"Corrected": 0.1, "Original": 0.3}
    assert opera.parse_acapella("{A: [], B: {}, C: two words}") == {"A": [], "B": {}, "C": "two words"}
    with pytest.raises(ValueError):
        opera.parse_acapella("{A: [1, 2")


def test_flatfield_profile_and_formula():
    ff = OperaFlatfield.from_profile(PROFILE)
    p = ff.profile((NY, NX))
    assert p.shape == (NY, NX)
    # P grows along columns (u) and decreases along rows (v)
    assert p[0, -1] > p[0, 0] and p[-1, 0] < p[0, 0]
    raw = raw_plane(1, 1, 1)
    corrected = np.rint(ff(raw))
    assert np.array_equal(corrected, expected_flatfield(raw))
    background = 100 * p
    assert (raw < background).any() and (raw >= background).any()  # both branches are tested
    # profile of another binning: same values at the same place in the field of view
    assert np.allclose(ff.profile((2 * NY, 2 * NX))[::2, ::2][1:, 1:], p[1:, 1:], atol=0.02)
    assert OperaFlatfield.from_profile(IDENTITY) is None
    # picklable without its cache
    clone = pickle.loads(pickle.dumps(ff))
    assert clone._cache == {} and np.array_equal(clone(raw), ff(raw))


def test_flatfield_foreground_is_ignored_with_a_warning(caplog):
    text = PROFILE.replace("Channel: 1,", "Foreground: {Character: NonFlat, Mean: 5.0, Profile: {Coefficients: "
                           "[[1.0]], Type: Polynomial}}, Channel: 1,")
    with caplog.at_level("WARNING", logger="pymif"):
        ff = OperaFlatfield.from_profile(text, "EGFP")
    assert ff.mean == 100.0 and "Foreground" in caplog.text


@pytest.mark.parametrize("block", [64, 1 << 24])  # tiny blocks cut the index anywhere
def test_read_index(export, monkeypatch, block):
    monkeypatch.setattr(opera, "_READ_BLOCK", block)
    channels, stacks = opera.read_index(export / "Images" / "Index.xml", progress=False)
    assert channels[1]["ChannelName"][0] == "EGFP" and "FlatfieldProfile" in channels[2]
    assert channels[1]["ImageResolutionX"] == ("1E-06", "m")
    assert len(stacks) == len(FIELDS) * 2
    s = stacks[(1, 1, 2, 1, 1)]
    assert s["x"] == pytest.approx(4.0) and s["y"] == 0.0
    assert [(p, round(z, 6)) for p, z, _ in s["planes"]] == [(1, -2.0), (2, 0.0), (3, 2.0)]
    assert stacks[(1, 1, 3, 1, 1)]["planes"][1][2] is None  # failed plane


def test_opera_scan(export):
    m = OperaMosaicManager(export, progress=False)
    assert m.channels == ["EGFP", "Brightfield"]
    assert len(m.tiles) == len(FIELDS) * 2
    assert m.fields == ["A1_field0", "A1_field1", "B1_field0"]
    assert sorted(t.attrs["opera_field"] for t in m.field_tiles("A1_field0") if t.channel == "EGFP") == [1, 2, 4]
    assert m.tiles_by_ids(["r01c01f02"]) == m.select(well="A1", opera_field=2)

    t = m.select(channel="EGFP", well="A1", opera_field=1)[0]
    assert m.tile_id(t) == "EGFP/r01c01f01"
    assert t.shape == (3, NY, NX) and t.dtype == np.uint16
    # x/y: resolution x OrientationMatrix scale; z: PositionZ step (no matrix scale or offset)
    assert t.voxel_size == (2.0, 0.5, 0.5)
    assert t.origin == pytest.approx((-2.0, -(NY - 1) / 2 * 0.5, -(NX - 1) / 2 * 0.5))
    right = m.select(channel="EGFP", well="A1", opera_field=2)[0]
    assert right.origin[2] - t.origin[2] == pytest.approx(4.0)
    up = m.select(channel="EGFP", well="A1", opera_field=4)[0]  # stage +y is up in the image
    assert up.origin[1] - t.origin[1] == pytest.approx(-3.0)
    # same stage position in well B1: one 96-well pitch (9 mm) further along y
    other_well = m.select(channel="EGFP", well="B1")[0]
    assert other_well.field == "B1_field0"
    assert np.subtract(other_well.origin, t.origin) == pytest.approx((0.0, 9000.0, 0.0))

    # flat-field: EGFP corrected as Harmony does, Brightfield (Identity) unchanged
    assert t.attrs["flatfield"] and isinstance(m.flatfields["EGFP"], OperaFlatfield)
    assert m.flatfields["Brightfield"] is None
    assert np.array_equal(t.data[1], expected_flatfield(raw_plane(1, 2, 1)))
    bf = m.select(channel="Brightfield", well="A1", opera_field=1)[0]
    assert np.array_equal(bf.data[:, 2:5, 3], np.stack([raw_plane(1, p, 2)[2:5, 3] for p in (1, 2, 3)]))

    failed = m.select(channel="EGFP", well="A1", opera_field=3)[0]
    assert not failed.data[1].any() and failed.data[0].any()


def test_opera_without_flatfield(export):
    m = OperaMosaicManager(export / "Images" / "Index.xml", flatfield=False, progress=False)
    t = m.select(channel="EGFP", well="A1", opera_field=1)[0]
    assert not t.attrs["flatfield"] and m.flatfields["EGFP"] is None
    assert np.array_equal(t.data[0], raw_plane(1, 1, 1))


@pytest.mark.parametrize("plate", [None, (7, 7), (16, 24)])
def test_well_pitch(tmp_path, plate):
    """Missing or unknown plate format: wells side by side, 10% apart; else the format's pitch."""
    assert opera.read_plate(write_export(tmp_path / "plate", plate=plate) / "Images" / "Index.xml") == plate
    m = OperaMosaicManager(tmp_path / "plate", progress=False)
    a1, b1 = (m.select(channel="EGFP", well=w, opera_field=1)[0] for w in ("A1", "B1"))
    pitch = 1.1 * (1000.0 + NX * 0.5)  # A1 fields span 1 mm along x, plus one tile
    if plate == (16, 24):  # 4.5 mm: wide enough
        pitch = 4500.0
    assert np.subtract(b1.origin, a1.origin) == pytest.approx((0.0, pitch, 0.0))


def test_flatfield_profiles_from_ffc_folder(tmp_path):
    root = write_export(tmp_path / "plate", flatfield_in_index=False)
    (root / "FFC_Profile").mkdir()
    (root / "FFC_Profile" / "FFC_Profile_Measurement 1.xml").write_text(
        '<Results xmlns="http://www.perkinelmer.com/PEHH/HarmonyV6"><Map>'
        f'<Entry ChannelID="1"><FlatfieldProfile>{PROFILE}</FlatfieldProfile></Entry>'
        f'<Entry ChannelID="2"><FlatfieldProfile>{IDENTITY}</FlatfieldProfile></Entry></Map></Results>')
    m = OperaMosaicManager(root / "Images", progress=False)
    t = m.select(channel="EGFP", well="A1", opera_field=1)[0]
    assert np.array_equal(t.data[0], expected_flatfield(raw_plane(1, 1, 1)))


def test_tiff_stack_array(tmp_path):
    for z in range(3):
        tifffile.imwrite(tmp_path / f"p{z}.tif", np.full((4, 5), z + 1, np.uint16))
    a = TiffStackArray(["p0.tif", None, "p2.tif"], (3, 4, 5), np.uint16, folder=tmp_path,
                       transform=lambda plane: plane * 1.5)
    assert a.chunks == (1, 4, 5) and len(a) == 3
    assert a[0].shape == (4, 5) and a[0][0, 0] == 2  # 1.5 rounded
    assert a[1:, 1:3, ::2].shape == (2, 2, 3) and not a[1].any()
    assert a[..., 0].shape == (3, 4) and a[2, 3, 4] == 4  # 4.5 rounds to even
    assert a[[0, 2]].shape == (2, 4, 5)
    with pytest.raises(ValueError):
        TiffStackArray(["p0.tif"], (2, 4, 5), np.uint16)

    # a plane read ahead is used by the next read of that plane only, and never pickled
    plain = TiffStackArray(["p0.tif"], (1, 4, 5), np.uint16, folder=tmp_path)
    plain.prefetch(0)
    assert pickle.loads(pickle.dumps(plain))._ready is None and plain._ready is not None
    reads = []
    read = a._read
    a._read = lambda z: reads.append(z) or read(z)
    a.prefetch(2)
    assert a[2, 0, 0] == 4 and a[2, 0, 0] == 4 and reads == [2, 2]
    a.prefetch(0)
    assert a[2, 0, 0] == 4 and a[0, 0, 0] == 2 and reads == [2, 2, 0, 2, 0]  # another plane drops it


def test_opera_visualize_reads_one_plane_per_tile(export, monkeypatch):
    """Tiles are read once, at the shown plane (napari would also read z = 0), ahead in threads."""
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    reads = []
    read = TiffStackArray._read
    monkeypatch.setattr(TiffStackArray, "_read", lambda self, z: reads.append((self.paths[0], z)) or read(self, z))
    m = OperaMosaicManager(export, progress=False)
    tiles = m.select(channel="EGFP", well="A1")
    v = ViewerModel()
    m.visualize(tiles=tiles, viewer=v)
    assert len(v.layers) == len(tiles) and all(layer.visible for layer in v.layers)
    assert sorted(reads) == sorted((t.data.paths[0], 1) for t in tiles)  # the middle of the 3 planes
    assert all(t.data._ready is None for t in tiles)  # handed over to napari
    assert v.layers[0].data[1].compute().any()


def test_opera_fuse_and_pickle(export):
    m = OperaMosaicManager(export, progress=False)
    t = m.select(channel="EGFP", well="A1", opera_field=1)[0]
    clone = pickle.loads(pickle.dumps(t.data))
    assert np.array_equal(clone[:], t.data[:])

    zm = m.fuse(export.parent / "a1.ome.zarr", field="A1_field0", projection="max", blend="max",
                progress=False, drop_singleton=False)
    data = np.asarray(zm.data[0])
    # tiles of fields 1, 2 and 4: 3 + 1 tile heights/widths minus the overlaps, 0.5 um pixels
    assert data.shape[-2:] == (NY + 6, NX + 8)
    assert zm.metadata["scales"][0][-2:] == (0.5, 0.5)
    with pytest.raises(Exception):
        m.fuse(export.parent / "both.ome.zarr", field=["A1_field0", "B1_field0"], progress=False)


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["pymif", *argv])
    cli.main()


def test_cli_fuse_opera(export, tmp_path, monkeypatch):
    assert cli._resolve_mosaic_manager(str(export))[1] == "opera"
    assert cli._resolve_mosaic_manager(str(export / "Images"))[1] == "opera"
    out, raw = tmp_path / "a1.ome.zarr", tmp_path / "a1_raw.ome.zarr"
    _run(monkeypatch, "fuse", "-i", str(export), "-z", str(out), "-p", "max", "-b", "max", "-f", "A1_field1")
    _run(monkeypatch, "fuse", "-i", str(export), "-z", str(raw), "-p", "max", "-b", "max", "-f", "A1_field1",
         "-nff")
    corrected = np.asarray(mm.ZarrManager(str(out)).data[0])
    plain = np.asarray(mm.ZarrManager(str(raw)).data[0])
    assert corrected.shape == plain.shape and not np.array_equal(corrected, plain)
    # field 3 has a failed plane: the projection of the two other planes
    assert np.array_equal(plain[0], np.maximum(raw_plane(3, 1, 1), raw_plane(3, 3, 1)))


def test_wrong_reader_is_refused(export, monkeypatch):
    """An Opera export opened with the Luxendo reader fails at once, naming the right reader."""
    from pymif.mosaic_manager import LuxendoMosaicManager, WrongReaderError, detect_microscope

    for path in (export, export / "Images", export / "Images" / "Index.xml"):
        assert detect_microscope(path) == "opera"
    assert detect_microscope(export.parent) is None
    monkeypatch.setattr(LuxendoMosaicManager, "_scan", lambda self: pytest.fail("scanned with the wrong reader"))
    with pytest.raises(WrongReaderError, match="OperaMosaicManager") as e:
        LuxendoMosaicManager(export)
    assert e.value.detected == "opera"
    with pytest.raises(WrongReaderError):
        _run(monkeypatch, "fuse", "-i", str(export), "-z", str(export.parent / "x.ome.zarr"), "-m", "luxendo")


def test_cli_fuse_batch_flatfield_column(export, tmp_path, monkeypatch):
    csv = tmp_path / "batch.csv"
    rows = [{"input": str(export), "microscope": "opera", "output": str(tmp_path / f"{name}.ome.zarr"),
             "projection": "max", "blend": "max", "fields": "A1_field1", "channels": "EGFP", "flatfield": ff}
            for name, ff in (("on", ""), ("off", "false"))]
    pd.DataFrame(rows).to_csv(csv, index=False)
    _run(monkeypatch, "fuse", "--batch", str(csv))
    on = np.asarray(mm.ZarrManager(str(tmp_path / "on.ome.zarr")).data[0])
    off = np.asarray(mm.ZarrManager(str(tmp_path / "off.ome.zarr")).data[0])
    assert np.array_equal(off, np.maximum(raw_plane(3, 1, 1), raw_plane(3, 3, 1)))
    assert not np.array_equal(on, off)


def test_reader_options():
    from pymif.mosaic_manager import LuxendoMosaicManager, reader_options

    assert reader_options(OperaMosaicManager, flatfield=False) == {"flatfield": False}
    assert reader_options(LuxendoMosaicManager, flatfield=False) == {}
