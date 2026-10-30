from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import logging
import os

import dask.array as da
import zarr

from .microscope_manager import MicroscopeManager
from ..utils.axes import normalize_axes, normalize_data_type
from ..utils.ngff import _infer_data_type_from_group, _register_label_on_labels_group
from collections.abc import Iterator, Sequence

logger = logging.getLogger("pymif")

if TYPE_CHECKING:
    import napari

from dataclasses import dataclass

@dataclass
class ZarrDataset(Sequence):
    data: List[da.Array]
    zarr_data: List[Any] | None
    metadata: Dict[str, Any]
    name: str | None = None
    path: str | None = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self) -> Iterator[da.Array]:
        return iter(self.data)

    def __repr__(self):
        return (
            f"ZarrDataset("
            f"name={self.name!r}, "
            f"levels={len(self.data)}, "
            f"shape={self.data[0].shape if self.data else None}, "
            f"dtype={self.data[0].dtype if self.data else None}, "
            f"axes={self.metadata.get('axes')!r}"
            f")"
        )

class AttrDict(dict):
    """Dictionary with optional attribute access for valid Python names."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

class ZarrManager(MicroscopeManager):
    """
    A manager class for reading and handling OME-Zarr datasets.

    Supports both:
    - NGFF / OME-Zarr v0.4  -> metadata in root/group attrs directly
    - NGFF / OME-Zarr v0.5  -> metadata in attrs["ome"]
    """

    def __init__(
        self,
        path,
        chunks: Tuple[int, ...] = None,
        mode: str = "r",
        metadata: dict[str, Any] = None,
        ngff_version: str | None = None,
        zarr_format: int | None = None,
    ):
        """Open or create an OME-Zarr dataset.

        Parameters
        ----------
        path : str or path-like
            Path to the zarr root directory.
        chunks : tuple[int, ...] | None
            Optional dask chunking to use when lazily reopening arrays.
        mode : {"r", "r+", "a", "w"}
            Open mode. ``"r"`` reads an existing dataset, ``"a"`` opens or
            creates one, and ``"w"`` creates a new root from metadata.
        metadata : dict | None
            Metadata required when creating an empty dataset. Ignored when
            reading an existing dataset.
        ngff_version, zarr_format : optional
            Explicit format override used when creating a new store.
        """
        super().__init__()
        self.path = path
        self.chunks = chunks
        self.mode = mode
        self.metadata = metadata
        self.ngff_version: str | None = None
        self.ngff_version_override = ngff_version
        self.zarr_format_override = zarr_format

        if os.path.exists(self.path):
            if mode in ("r", "a", "r+"):
                self.root = zarr.open(zarr.storage.LocalStore(self.path), mode=self.mode)
                self.read()
            else:
                raise FileNotFoundError(
                    f"Zarr path {self.path} exists and mode='{mode}' is write-only. "
                    "Please select mode='r', 'r+' or 'a'."
                )
        else:
            if mode in ("w", "a"):
                from ..utils.ngff import ZarrWriteConfig, _resolve_format

                cfg = ZarrWriteConfig(
                    ngff_version=self.ngff_version_override or (self.metadata or {}).get("ngff_version"),
                    zarr_format=self.zarr_format_override or (self.metadata or {}).get("zarr_format"),
                )
                resolved_ngff_version, resolved_zarr_format = _resolve_format(cfg)

                self.root = zarr.open_group(
                    zarr.storage.LocalStore(self.path),
                    mode=self.mode,
                    zarr_format=resolved_zarr_format,
                )

                from ..utils.create_empty_dataset import create_empty_dataset as _create_empty_dataset
                _create_empty_dataset(
                    self.root,
                    self.metadata,
                    ngff_version=resolved_ngff_version,
                    zarr_format=resolved_zarr_format,
                )
                self.read()
            else:
                raise FileNotFoundError(
                    f"Zarr path {self.path} does not exist and mode='{mode}' is read-only. "
                    "Please select mode='w' or 'a'."
                )

    def _get_image_meta(self, group: zarr.Group) -> dict[str, Any]:
        """
        Return the image metadata dictionary for either NGFF v0.4 or v0.5.

        v0.5 -> group.attrs["ome"]
        v0.4 -> group.attrs
        """
        attrs = group.attrs.asdict()
        if "ome" in attrs and isinstance(attrs["ome"], dict):
            self.ngff_version = attrs["ome"].get("version", "0.5")
            return attrs["ome"]

        self.ngff_version = "0.4"
        return attrs

    def _get_multiscales(self, group: zarr.Group) -> list[dict[str, Any]]:
        """Return the NGFF ``multiscales`` block for ``group`` regardless of version."""
        image_meta = self._get_image_meta(group)
        return image_meta.get("multiscales", [])

    def _get_omero(self, group: zarr.Group) -> dict[str, Any]:
        """Return the OMERO-style channel metadata stored on ``group``."""
        image_meta = self._get_image_meta(group)
        return image_meta.get("omero", {})

    def _extract_metadata(
        self,
        data_levels: List[da.Array],
        datasets: list[dict[str, Any]],
        multiscales: dict[str, Any],
        omero: dict[str, Any],
        data_type: str = "intensity",
        zarr_levels: List[Any] | None = None,
    ) -> Dict[str, Any]:
        """Translate NGFF metadata blocks into the normalized PyMIF schema.

        The normalized schema is axis-aware: ``axes`` may be any unique
        combination of ``t``, ``c``, ``z``, ``y`` and ``x``. Spatial ``scales``
        and ``units`` are stored in the same order as the spatial axes appear in
        ``axes``.
        """
        axes_info = multiscales.get("axes", [])
        axis_names = []
        for axis in axes_info:
            if isinstance(axis, dict):
                axis_names.append(axis.get("name"))
            else:
                axis_names.append(axis)
        axis_names = list(normalize_axes(axis_names, ndim=data_levels[0].ndim))
        axes_info = axes_info if len(axes_info) == len(axis_names) else [{"name": ax} for ax in axis_names]
        axes = "".join(axis_names)
        data_type = normalize_data_type(data_type)

        sizes = [tuple(arr.shape) for arr in data_levels]
        chunksize = [arr.chunksize for arr in data_levels]
        dtype = data_levels[0].dtype

        shards = None
        if zarr_levels is not None:
            shard_shapes = [getattr(z, "shards", None) for z in zarr_levels]
            if any(s is not None for s in shard_shapes):
                shards = [tuple(s) if s is not None else None for s in shard_shapes]

        spatial_idx = [i for i, ax in enumerate(axis_names) if ax in ("z", "y", "x")]
        time_idx = axis_names.index("t") if "t" in axis_names else None
        channel_idx = axis_names.index("c") if "c" in axis_names else None

        scales = []
        for ds in datasets:
            ct = ds.get("coordinateTransformations", [{}])
            scale_vec = None
            if ct and isinstance(ct, list):
                for transform in ct:
                    if isinstance(transform, dict) and transform.get("type", "scale") == "scale":
                        scale_vec = transform.get("scale", None)
                        break
            if scale_vec is None:
                scales.append(tuple([1.0] * len(spatial_idx)))
            else:
                scales.append(tuple(float(scale_vec[i]) for i in spatial_idx))

        units = tuple(
            axes_info[i].get("unit", None) if isinstance(axes_info[i], dict) else None
            for i in spatial_idx
        )

        if time_idx is not None:
            ct0 = datasets[0].get("coordinateTransformations", [{}])
            scale0 = None
            if ct0 and isinstance(ct0, list):
                for transform in ct0:
                    if isinstance(transform, dict) and transform.get("type", "scale") == "scale":
                        scale0 = transform.get("scale", None)
                        break
            time_increment = scale0[time_idx] if scale0 is not None else None
            time_increment_unit = (
                axes_info[time_idx].get("unit", None)
                if isinstance(axes_info[time_idx], dict)
                else None
            )
        else:
            time_increment = None
            time_increment_unit = None

        if channel_idx is not None and data_type == "intensity":
            channels = omero.get("channels", [])
            c_size = data_levels[0].shape[channel_idx]
            channel_names = [
                channels[i].get("label", f"Channel {i}") if i < len(channels) else f"Channel {i}"
                for i in range(c_size)
            ]
            channel_colors = [
                channels[i].get("color", "FFFFFF") if i < len(channels) else "FFFFFF"
                for i in range(c_size)
            ]
        else:
            channel_names = []
            channel_colors = []

        return {
            "size": sizes,
            "chunksize": chunksize,
            "shards": shards,
            "scales": scales,
            "units": units,
            "time_increment": time_increment,
            "time_increment_unit": time_increment_unit,
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "dtype": str(dtype),
            "plane_files": None,
            "axes": axes,
            "data_type": data_type,
            "is_label": data_type == "label",
            "ngff_version": self.ngff_version,
            "zarr_format": 3 if self.ngff_version == "0.5" else 2,
        }

    def _read_multiscale_group(
        self,
        group: zarr.Group,
    ) -> tuple[List[da.Array], List[Any], Dict[str, Any]]:
        """Load one multiscale image group as dask arrays plus normalized metadata."""
        multiscales_all = self._get_multiscales(group)
        if not multiscales_all:
            raise ValueError(f"Group '{group.name}' does not contain multiscales metadata.")

        multiscales = multiscales_all[0]
        datasets = multiscales.get("datasets", [])
        if not datasets:
            raise ValueError(f"Group '{group.name}' has multiscales metadata but no datasets.")

        omero = self._get_omero(group)

        data_levels = []
        zarr_levels = []

        for ds in datasets:
            path = ds["path"]
            zarr_array = group[path]
            zarr_levels.append(zarr_array)

            if self.chunks is not None and len(self.chunks) == len(zarr_array.shape):
                arr = da.from_zarr(zarr_array, chunks=self.chunks)
            else:
                arr = da.from_zarr(zarr_array)

            data_levels.append(arr)

        metadata = self._extract_metadata(
            data_levels=data_levels,
            datasets=datasets,
            multiscales=multiscales,
            omero=omero,
            data_type=_infer_data_type_from_group(group),
            zarr_levels=zarr_levels,
        )

        return data_levels, zarr_levels, metadata

    def read(self) -> None:
        """Read the root image plus discover additional image groups and labels.

        The root image is exposed through ``self.data`` and ``self.metadata``.
        Additional multiscale subgroups are indexed in ``self.groups`` and label
        pyramids in ``self.labels``.
        """
        data_levels, zarr_levels, metadata = self._read_multiscale_group(self.root)

        self.raw = ZarrDataset(
            data=data_levels,
            zarr_data=zarr_levels,
            metadata=metadata,
            name="raw",
            path="/",
        )

        # Backward-compatible aliases
        self._sync_raw_aliases()
        self.chunks = data_levels[0].chunksize

        self.groups = AttrDict()
        self.labels = AttrDict()

        for name in self.root.group_keys():
            if name == "labels":
                self.labels = self._load_labels()
            else:
                group_dataset = self._load_group(name)

                if group_dataset is not None:
                    self.groups[name] = group_dataset

                    # Optional convenience: d.proc.data
                    if name.isidentifier() and not hasattr(self, name):
                        setattr(self, name, group_dataset)

        logger.info("%s", self.root.tree())
        for k, v in self.metadata.items():
            logger.info("%s: %s", k.upper(), v)

    # ``read()`` is overridden because a Zarr store holds several datasets (root
    # image, groups, labels). The template hooks expose the already-read root.
    def _parse_metadata(self) -> Dict[str, Any]:
        return self.raw.metadata

    def _build_dask_array(self) -> List[da.Array]:
        return self.raw.data

    def _sync_raw_aliases(self) -> None:
        """
        Keep the old single-dataset API synchronized with the raw dataset.

        This preserves:
            d.data
            d.metadata
            d.zarr_data

        as aliases to:
            d.raw.data
            d.raw.metadata
            d.raw.zarr_data
        """
        self.data = self.raw.data
        self.metadata = self.raw.metadata
        self.zarr_data = self.raw.zarr_data

    def _iter_datasets(
        self,
        include_raw: bool = True,
        include_groups: bool = True,
        include_labels: bool = True,
    ):
        """
        Iterate over all datasets managed by ZarrManager.

        This is ZarrManager-specific because only ZarrManager knows about
        raw/groups/labels.
        """
        if include_raw and hasattr(self, "raw"):
            yield "raw", self.raw

        if include_groups and hasattr(self, "groups"):
            for name, dataset in self.groups.items():
                yield name, dataset

        if include_labels and hasattr(self, "labels"):
            for name, dataset in self.labels.items():
                yield f"labels/{name}", dataset

    def _invalidate_zarr_data(self, dataset: ZarrDataset) -> None:
        """
        After lazy in-memory transformations, dataset.data no longer necessarily
        corresponds one-to-one to the original on-disk Zarr arrays.
        """
        dataset.zarr_data = None

    def _load_group(self, name):
        group = self.root[name]

        try:
            arrays, zarr_arrays, metadata = self._read_multiscale_group(group)
        except ValueError:
            return None

        metadata["is_label"] = False
        metadata["name"] = name

        return ZarrDataset(
            data=arrays,
            zarr_data=zarr_arrays,
            metadata=metadata,
            name=name,
            path=name,
        )

    def _load_labels(self):
        labels = AttrDict()

        if "labels" not in self.root:
            return labels

        labels_grp = self.root["labels"]

        for label_name, label_grp in labels_grp.groups():
            try:
                arrays, zarr_arrays, metadata = self._read_multiscale_group(label_grp)
            except ValueError:
                continue

            metadata = dict(metadata)
            metadata["is_label"] = True
            metadata["name"] = label_name

            labels[label_name] = ZarrDataset(
                data=arrays,
                zarr_data=zarr_arrays,
                metadata=metadata,
                name=label_name,
                path=f"labels/{label_name}",
            )

        return labels

    def visualize_zarr(
        self,
        viewer: "napari.Viewer | None " = None,
    ) -> "napari.Viewer | None":
        """Open the root image and all multiscale subgroups in napari."""
        try:
            import napari
        except ImportError:
            import warnings
            warnings.warn(
                "napari is not installed. Install with `pip install pymif[napari]` to use visualization.",
                stacklevel=2,
            )
            return None

        if viewer is None:
            viewer = napari.Viewer()

        def discover_multiscales(group: zarr.Group, path=""):
            results = []

            try:
                multiscales = self._get_multiscales(group)
                if multiscales:
                    results.append(path)
            except Exception:
                pass

            for name, sub in group.groups():
                if name != "labels":
                    subpath = f"{path}/{name}" if path else name
                    results.extend(discover_multiscales(sub, path=subpath))

            return results

        multiscale_paths = discover_multiscales(self.root)

        for gpath in multiscale_paths:
            full_path = Path(self.path) / gpath
            viewer.open(full_path, plugin="napari-ome-zarr")

        return viewer

    def create_empty_group(
        self,
        group_name: str,
        metadata: Dict[str, Any],
        is_label: bool = False,
        data_type: str | None = None,
    ):
        """Create an empty image subgroup or label subgroup and update this manager.

        ``metadata['axes']`` may be any unique subset of ``tczyx``. Use
        ``data_type='label'`` or ``is_label=True`` for label data.
        """
        from ..utils.create_empty_group import create_empty_group as _create_empty_group

        effective_data_type = normalize_data_type(data_type or metadata.get("data_type"), is_label=True if is_label else None)
        is_label = is_label or effective_data_type == "label"

        grp = _create_empty_group(
            root=self.root,
            group_name=group_name,
            metadata=metadata,
            is_label=is_label,
            data_type=data_type,
        )

        # Read the newly created group back into the in-memory ZarrDataset model.
        if is_label:
            arrays, zarr_arrays, group_metadata = self._read_multiscale_group(grp)

            group_metadata = dict(group_metadata)
            group_metadata["is_label"] = True
            group_metadata["name"] = group_name

            dataset = ZarrDataset(
                data=arrays,
                zarr_data=zarr_arrays,
                metadata=group_metadata,
                name=group_name,
                path=f"labels/{group_name}",
            )

            if not hasattr(self, "labels"):
                self.labels = AttrDict()

            self.labels[group_name] = dataset

        else:
            arrays, zarr_arrays, group_metadata = self._read_multiscale_group(grp)

            group_metadata = dict(group_metadata)
            group_metadata["is_label"] = False
            group_metadata["name"] = group_name

            dataset = ZarrDataset(
                data=arrays,
                zarr_data=zarr_arrays,
                metadata=group_metadata,
                name=group_name,
                path=group_name,
            )

            if not hasattr(self, "groups"):
                self.groups = AttrDict()

            self.groups[group_name] = dataset

            # Optional convenience: d.proc.data
            if group_name.isidentifier():
                setattr(self, group_name, dataset)

        return dataset

    def write_image_region(
        self,
        data,
        t: int | slice = slice(None),
        c: int | slice = slice(None),
        z: int | slice = slice(None),
        y: int | slice = slice(None),
        x: int | slice = slice(None),
        level: int = 0,
        group: Optional[str] = None,
        downscale_factor: int | Sequence[int] | None = None,
    ):
        """Write an image patch into a root or subgroup pyramid and refresh lower levels."""
        from ..utils.write_image_region import write_image_region as _write_image_region
        return _write_image_region(
            root=self.root,
            mode=self.mode,
            data=data,
            t=t,
            c=c,
            z=z,
            y=y,
            x=x,
            level=level,
            group_name=group,
            downscale_factor=downscale_factor,
        )

    def write_label_region(
        self,
        data,
        t: int | slice = slice(None),
        z: int | slice = slice(None),
        y: int | slice = slice(None),
        x: int | slice = slice(None),
        level: int = 0,
        group: str = None,
        downscale_factor: int | Sequence[int] | None = None,
    ):
        """Write a label patch into a label pyramid and regenerate coarser levels."""
        from ..utils.write_label_region import write_label_region as _write_label_region
        return _write_label_region(
            root=self.root,
            mode=self.mode,
            data=data,
            t=t,
            z=z,
            y=y,
            x=x,
            level=level,
            group_name=group,
            downscale_factor=downscale_factor,
        )
    
    def subset_dataset(
        self,
        T=None,
        C=None,
        Z=None,
        Y=None,
        X=None,
        include_groups: bool = True,
        include_labels: bool = True,
        rebuild_pyramid=False
    ):
        """
        Subset raw, groups, and labels.

        Channel subsetting is skipped automatically for datasets without a
        channel axis, such as most label datasets.
        """
        from ..utils.dataset_ops import subset_levels

        for name, dataset in self._iter_datasets(
            include_raw=True,
            include_groups=include_groups,
            include_labels=include_labels,
        ):
            if not dataset.data:
                continue

            dataset.data, dataset.metadata = subset_levels(
                dataset.data,
                dataset.metadata,
                T=T, C=C, Z=Z, Y=Y, X=X,
                rebuild_pyramid=rebuild_pyramid,
                label=f"dataset {name!r}",
            )
            self._invalidate_zarr_data(dataset)

        self._sync_raw_aliases()

        logger.info("Zarr datasets subset complete.")

    def build_pyramid(
        self,
        num_levels: Optional[int] = 3,
        downscale_factor: int | Sequence[int] | None = 2,
        start_level: int = 0,
        include_groups: bool = True,
        include_labels: bool = True,
    ):
        """
        Build/rebuild pyramids for raw, groups, and labels.
        """
        from ..utils.pyramid import build_pyramid as _build_pyramid

        for name, dataset in self._iter_datasets(
            include_raw=True,
            include_groups=include_groups,
            include_labels=include_labels,
        ):
            if not dataset.data:
                continue

            dataset.data, dataset.metadata = _build_pyramid(
                dataset.data,
                dataset.metadata,
                num_levels=num_levels,
                downscale_factor=downscale_factor,
                start_level=start_level,
            )

            self._invalidate_zarr_data(dataset)

        self._sync_raw_aliases()

        logger.info("Zarr pyramids rebuilt.")

    def reorder_channels(
        self,
        new_order: List[int],
        include_groups: bool = True,
    ):
        """
        Reorder channels in raw and non-label image groups.

        Labels are skipped because they usually do not have a channel axis.
        """
        from ..utils.dataset_ops import reorder_channels as _reorder_channels

        for name, dataset in self._iter_datasets(
            include_raw=True,
            include_groups=include_groups,
            include_labels=False,
        ):
            if not dataset.data:
                continue

            dataset.data = _reorder_channels(
                dataset.data,
                dataset.metadata,
                new_order,
                label=f"dataset {name!r}",
                skip_if_no_channel_axis=True,
            )
            self._invalidate_zarr_data(dataset)

        self._sync_raw_aliases()

        logger.info("Channels reordered to %s.", list(new_order))

    def estimate_chromatic_shift(
        self,
        reference_channel: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Estimate the axial (Z) chromatic shift of each channel of the raw image.

        The shift is measured on the sample's Z-facing surfaces (top and
        bottom), first on a coarse level and then on crops of a finer level.
        Keyword arguments are forwarded to
        :func:`~pymif.utils.chromatic_shift.estimate_chromatic_shift`
        (``timepoint``, ``coarse_level``, ``refine_level``, ``max_shift``,
        ``n_regions``, ``crop_yx``, ``min_z_normal``, ``min_correlation``,
        ``threshold``).

        Returns a JSON-serializable report whose ``shifts`` entry holds the
        correction per channel in the physical unit of the Z axis; it can be
        passed directly to :meth:`apply_chromatic_shift`.
        """
        from ..utils.chromatic_shift import estimate_chromatic_shift as _estimate

        return _estimate(
            self.raw.data,
            self.raw.metadata,
            reference_channel=reference_channel,
            **kwargs,
        )

    def apply_chromatic_shift(
        self,
        path: str | Path,
        shifts,
        *,
        extent: str = "union",
        order: int = 1,
        include_groups: bool = True,
        include_labels: bool = True,
        overwrite: bool = True,
        num_workers: int | None = None,
        progress: bool = True,
    ) -> "ZarrManager":
        """Write a copy of this store with each channel shifted along Z.

        Parameters
        ----------
        path
            Output OME-Zarr store; must differ from the input.
        shifts
            One Z shift per channel in physical units (positive moves content
            towards higher Z), or the report of :meth:`estimate_chromatic_shift`.
        extent
            ``"union"`` (default) grows Z so no channel loses data,
            ``"same"`` keeps the input size and ``"intersection"`` crops to
            the planes covered by every channel. Padding is rounded to whole
            planes of the coarsest level so all pyramid levels stay aligned.
        order
            ``1`` interpolates linearly; ``0`` rounds each shift to whole
            planes, which leaves voxel values untouched.
        include_groups, include_labels
            Also write image subgroups (shifted like the raw image) and labels
            (padded or cropped to the new Z extent, following the reference
            channel).
        overwrite, num_workers, progress
            Same meaning as in :class:`~pymif.utils.ngff.ZarrWriteConfig`.

        Every array keeps its source's chunks, shards, codecs and dimension
        names, every group its attributes (pixel sizes, channel metadata...).
        Datasets must still match their on-disk arrays, so call this before
        in-memory edits such as :meth:`subset_dataset`.

        Returns the output store opened as a new :class:`ZarrManager`.
        """
        from ..utils.chromatic_shift import write_chromatic_shift_corrected

        if Path(path).resolve() == Path(self.path).resolve():
            raise ValueError("apply_chromatic_shift needs an output path different from the input.")

        write_chromatic_shift_corrected(
            self.root,
            self._iter_datasets(
                include_raw=True,
                include_groups=include_groups,
                include_labels=include_labels,
            ),
            path,
            shifts,
            extent=extent,
            order=order,
            overwrite=overwrite,
            num_workers=num_workers,
            progress=progress,
        )
        return ZarrManager(str(path), chunks=None, mode="r")

    def correct_chromatic_shift(
        self,
        path: str | Path,
        reference_channel: int = 0,
        *,
        extent: str = "union",
        order: int = 1,
        include_groups: bool = True,
        include_labels: bool = True,
        overwrite: bool = True,
        num_workers: int | None = None,
        progress: bool = True,
        **estimate_kwargs,
    ) -> "ZarrManager":
        """Estimate the Z chromatic shift and write the corrected store in one go.

        Combines :meth:`estimate_chromatic_shift` (``estimate_kwargs``) and
        :meth:`apply_chromatic_shift`. The estimate report is stored in the
        output root attribute ``pymif_chromatic_shift``.
        """
        report = self.estimate_chromatic_shift(reference_channel, **estimate_kwargs)
        return self.apply_chromatic_shift(
            path,
            report,
            extent=extent,
            order=order,
            include_groups=include_groups,
            include_labels=include_labels,
            overwrite=overwrite,
            num_workers=num_workers,
            progress=progress,
        )

    def update_metadata(
        self,
        updates: Dict[str, Any],
        include_groups: bool = True,
        include_labels: bool = True,
    ):
        """
        Update metadata for raw, groups, and labels.

        Channel-specific metadata is skipped for datasets without a channel axis.
        """
        from ..utils.dataset_ops import apply_metadata_updates, known_updates

        updates = known_updates(updates)  # warns once about unknown keys

        for name, dataset in self._iter_datasets(
            include_raw=True,
            include_groups=include_groups,
            include_labels=include_labels,
        ):
            if not dataset.metadata:
                continue

            apply_metadata_updates(
                dataset.data,
                dataset.metadata,
                updates,
                label=f"dataset {name!r}",
                warn_no_channel_axis=False,  # e.g. labels have no channel axis
            )

        self._sync_raw_aliases()

        logger.info("Zarr metadata updated.")

    def to_zarr(
        self,
        path: str | Path,
        include_groups: bool = True,
        include_labels: bool = True,
        reread: bool = True,
        **kwargs,
    ):
        """Write raw data, groups, and labels to a complete OME-Zarr store.

        Raw data is written directly to the root group.
        Image groups are written as root-level subgroups.
        Labels are written under /labels.
        """
        from ..utils.to_zarr import write_multiscale_to_group
        from ..utils.ngff import ZarrWriteConfig, _resolve_format

        cfg = ZarrWriteConfig(**kwargs)
        ngff_version, zarr_format = _resolve_format(cfg)

        root = zarr.open_group(
            str(Path(path)),
            mode="w" if cfg.overwrite else "w-",
            zarr_format=zarr_format,
        )

        # ------------------------------------------------------------
        # 1. Raw dataset at root
        # ------------------------------------------------------------
        write_multiscale_to_group(
            group=root,
            data_levels=self.raw.data,
            metadata=self.raw.metadata,
            config=cfg,
            name=self.raw.name or "raw",
            is_label=False,
        )

        # ------------------------------------------------------------
        # 2. Non-label image groups at root/group_name
        # ------------------------------------------------------------
        if include_groups:
            for group_name, dataset in self.groups.items():
                group = root.require_group(group_name)

                write_multiscale_to_group(
                    group=group,
                    data_levels=dataset.data,
                    metadata=dataset.metadata,
                    config=cfg,
                    name=dataset.name or group_name,
                    is_label=False,
                )

        # ------------------------------------------------------------
        # 3. Label groups at root/labels/label_name
        # ------------------------------------------------------------
        if include_labels and self.labels:
            labels_group = root.require_group("labels")

            label_names = []

            for label_name, dataset in self.labels.items():
                label_names.append(label_name)

                label_group = labels_group.require_group(label_name)

                write_multiscale_to_group(
                    group=label_group,
                    data_levels=dataset.data,
                    metadata=dataset.metadata,
                    config=cfg,
                    name=dataset.name or label_name,
                    is_label=True,
                )

            # Label discovery metadata for the active NGFF layout.
            for name in label_names:
                _register_label_on_labels_group(root, name, ngff_version)

        self.path = str(path)
        self.root = root
        self.mode = "r+"
        self.ngff_version = ngff_version
        self.zarr_format_override = zarr_format

        if reread and cfg.compute:
            self.read()

        return root
