import dask.array as da

from pymif.utils.pyramid import build_pyramid
from pymif.utils.write_image_region import (
    _scale_index as _scale_image_index,
)
from pymif.utils.write_label_region import (
    _scale_index as _scale_label_index,
)


def test_anisotropic_build_pyramid_shapes_and_scales():
    data = da.ones((1, 1, 3, 8, 8), chunks=(1, 1, 3, 8, 8))
    metadata = {"scales": [(1.0, 1.0, 1.0)]}

    pyramid, metadata = build_pyramid(
        [data],
        metadata,
        num_levels=3,
        downscale_factor=(1, 2, 2),
    )

    assert [p.shape for p in pyramid] == [
        (1, 1, 3, 8, 8),
        (1, 1, 3, 4, 4),
        (1, 1, 3, 2, 2),
    ]

    assert metadata["scales"] == [
        (1.0, 1.0, 1.0),
        (1.0, 2.0, 2.0),
        (1.0, 4.0, 4.0),
    ]


def test_build_pyramid_keeps_chunk_size_level_independent():
    data = da.ones((1, 1, 3, 32, 32), chunks=(1, 1, 3, 8, 8))
    metadata = {"scales": [(1.0, 1.0, 1.0)]}

    pyramid, metadata = build_pyramid(
        [data],
        metadata,
        num_levels=3,
        downscale_factor=(1, 2, 2),
    )

    assert [p.shape for p in pyramid] == [
        (1, 1, 3, 32, 32),
        (1, 1, 3, 16, 16),
        (1, 1, 3, 8, 8),
    ]
    assert [p.chunksize for p in pyramid] == [
        (1, 1, 3, 8, 8),
        (1, 1, 3, 8, 8),
        (1, 1, 3, 8, 8),
    ]
    assert metadata["chunksize"] == [
        (1, 1, 3, 8, 8),
        (1, 1, 3, 8, 8),
        (1, 1, 3, 8, 8),
    ]


def test_scale_image_index_anisotropic():
    index = (
        0,
        0,
        slice(3, 4),
        slice(20, 40),
        slice(30, 50),
    )
    shape = (1, 1, 1, 10, 10)

    scaled = _scale_image_index(index, shape, (1, 2, 2))

    assert scaled == (
        0,
        0,
        slice(3, 4),
        slice(10, 20),
        slice(15, 25),
    )


def test_scale_label_index_anisotropic():
    index = (
        0,
        slice(3, 4),
        slice(20, 40),
        slice(30, 50),
    )
    shape = (1, 1, 10, 10)

    scaled = _scale_label_index(index, shape, (1, 2, 2))

    assert scaled == (
        0,
        slice(3, 4),
        slice(10, 20),
        slice(15, 25),
    )
