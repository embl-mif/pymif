try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - only when running from an unbuilt source tree
    __version__ = "0.0.0"

from . import microscope_manager
from . import cli

from . import mosaic_manager
