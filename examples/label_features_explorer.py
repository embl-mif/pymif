"""
Explore the features of segmented objects in napari, with napari-clusters-plotter.

Opens napari with the Objects Explorer widget of ZarrSmith (also in Plugins > ZarrSmith > Objects Explorer),
and loads the files given on the command line. See zarrsmith/napari/_explorer_widget.py for what it does.

Run:
    python label_features_explorer.py
    python label_features_explorer.py image.ome.zarr labels.ome.zarr features.csv
    python label_features_explorer.py image.ome.zarr "" features.csv --label-column label

Needs napari and napari-clusters-plotter (pip install zarrsmith[napari]).
"""

import argparse

import napari
from qtpy.QtCore import QTimer

from zarrsmith.napari._explorer_widget import explorer_widget


def main():
    parser = argparse.ArgumentParser(description="Explore the features of segmented objects in napari.")
    parser.add_argument("image", nargs="?", default="", help="intensity .ome.zarr")
    parser.add_argument("labels", nargs="?", default="", help="labels .ome.zarr ('' = <image>/labels/)")
    parser.add_argument("table", nargs="?", default="", help="features .csv")
    parser.add_argument("--label-column", help="column of the table with the label ids")
    parser.add_argument("--centroids-um", action="store_true", help="centroids of the table are in µm")
    args = parser.parse_args()

    viewer = napari.Viewer(title="Label features explorer")
    dock = explorer_widget()
    viewer.window.add_dock_widget(dock, name="Objects Explorer", area="right")

    explorer = dock.explorer
    explorer.image_field.value = args.image or None
    explorer.labels_field.value = args.labels or None
    explorer.table_field.value = args.table or None
    if args.label_column:
        explorer.label_column.setCurrentText(args.label_column)
    explorer.centroids_um.setChecked(args.centroids_um)
    if args.image and args.table:
        QTimer.singleShot(0, explorer.load)
    napari.run()


if __name__ == "__main__":
    main()
