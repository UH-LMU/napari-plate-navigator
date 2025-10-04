try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import make_qwidget

__all__ = ["make_qwidget"]
