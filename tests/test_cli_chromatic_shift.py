# tests/test_cli_chromatic_shift.py
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.cli import zarrsmith as cli
from zarrsmith.utils.chromatic_shift import PROVENANCE_KEY

from .test_chromatic_shift import SPACING, _expected_shifts, _synthetic_sample, _write_store


@pytest.fixture(scope="module")
def sample():
    return _synthetic_sample()


@pytest.fixture
def store(tmp_path, sample):
    _write_store(tmp_path / "in.ome.zarr", sample)
    return tmp_path / "in.ome.zarr"


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["zarrsmith", *map(str, argv)])
    cli.main()


def test_estimate_and_correct(store, tmp_path, monkeypatch, capsys):
    out, report = tmp_path / "out.ome.zarr", tmp_path / "shift.json"
    _run(monkeypatch, "chromatic-shift", "-i", store, "-z", out, "-r", report, "-mxs", 10)

    shifts = json.loads(report.read_text())["shifts"]
    np.testing.assert_allclose(shifts, _expected_shifts(), atol=0.25 * SPACING[0])
    z = mm.ZarrManager(str(out))
    assert z.root.attrs[PROVENANCE_KEY]["shifts"] == shifts
    assert z.metadata["size"][0][1] == 48 + 5  # union: 3 planes below, 2 above
    assert "ch1: -5." in capsys.readouterr().out


def test_measure_only(store, tmp_path, monkeypatch):
    report = tmp_path / "shift.json"
    _run(monkeypatch, "chromatic-shift", "-i", store, "-r", report, "-rl", "none")

    data = json.loads(report.read_text())
    assert data["refine_level"] is None and data["regions"] == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.ome.zarr", "shift.json"]


def test_given_shifts_and_report_file(store, tmp_path, monkeypatch, sample):
    out = tmp_path / "out.ome.zarr"
    _run(monkeypatch, "chromatic-shift", "-i", store, "-z", out, "-sf", 0, 4, -2, "-o", 0, "-e", "same")
    data = np.asarray(mm.ZarrManager(str(out)).data[0])
    np.testing.assert_array_equal(data[1, 2:], sample[1, :-2])
    np.testing.assert_array_equal(data[2, :-1], sample[2, 1:])

    report = tmp_path / "shift.json"
    report.write_text(json.dumps({"shifts": [0.0, 4.0, -2.0]}))
    out2 = tmp_path / "out2.ome.zarr"
    _run(monkeypatch, "chromatic-shift", "-i", store, "-z", out2, "-sf", report, "-o", 0, "-e", "same")
    np.testing.assert_array_equal(np.asarray(mm.ZarrManager(str(out2)).data[0]), data)


def test_reference_channel_by_name(store, tmp_path, monkeypatch):
    report = tmp_path / "shift.json"
    _run(monkeypatch, "chromatic-shift", "-i", store, "-r", report, "-rc", "ch1", "-mxs", 10)

    data = json.loads(report.read_text())
    assert data["reference_channel"] == 1 and data["shifts"][1] == 0.0


def test_existing_output_needs_overwrite(store, tmp_path, monkeypatch):
    out = tmp_path / "out.ome.zarr"
    out.mkdir()
    with pytest.raises(FileExistsError):
        _run(monkeypatch, "chromatic-shift", "-i", store, "-z", out, "-sf", 0, 0, 0)
    _run(monkeypatch, "chromatic-shift", "-i", store, "-z", out, "-sf", 0, 0, 0, "--overwrite")
    assert mm.ZarrManager(str(out)).metadata["size"][0] == (3, 48, 96, 96)


def test_fail_on_warning_writes_nothing(store, tmp_path, monkeypatch):
    out, report = tmp_path / "out.ome.zarr", tmp_path / "shift.json"
    with pytest.warns(UserWarning, match="search limit"), pytest.raises(SystemExit, match="has warnings"):
        _run(monkeypatch, "chromatic-shift", "-i", store, "-z", out, "-r", report,
             "-mxs", 1, "-rl", "none", "--fail_on_warning")

    assert not out.exists()
    assert json.loads(report.read_text())["warnings"]


def test_batch(store, tmp_path, monkeypatch):
    out = tmp_path / "out.ome.zarr"
    report_a, report_b = tmp_path / "a.json", tmp_path / "b.json"
    csv = tmp_path / "batch.csv"
    pd.DataFrame([
        {"input": store, "output": out, "report": report_a, "max_shift": 10, "order": 0, "extent": "same"},
        {"input": store, "output": "", "report": report_b, "max_shift": "", "refine_level": "none", "order": ""},
    ]).to_csv(csv, index=False)

    _run(monkeypatch, "chromatic-shift", "--batch", csv)

    assert mm.ZarrManager(str(out)).metadata["size"][0] == (3, 48, 96, 96)
    np.testing.assert_allclose(json.loads(report_a.read_text())["shifts"], _expected_shifts(),
                               atol=0.25 * SPACING[0])
    assert json.loads(report_b.read_text())["refine_level"] is None


def test_argument_errors(store, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run(monkeypatch, "chromatic-shift", "-z", tmp_path / "out.ome.zarr")  # no input
    with pytest.raises(SystemExit):
        _run(monkeypatch, "chromatic-shift", "--batch", store / "zarr.json", "-i", store)
