try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import make_qwidget

__all__ = ["make_qwidget"]

# In make_qwidget (or plugin entry)
import logging

LOG_TO_FILE = True  # Toggle here
LOG_FILE = "napari-plate-navigator.log"  # Or Path.home() / 'logs' / ...

logging.basicConfig(
    level=logging.INFO,  # DEBUG for verbose
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Includes module
    force=True,
    handlers=[logging.StreamHandler()],  # Always stdout
)

# Your package logger: DEBUG for your modules only
my_logger = logging.getLogger("napari_plate_navigator")
my_logger.setLevel(
    logging.DEBUG
)  # Enables DEBUG in _base, _widget, _reader, etc.

if LOG_TO_FILE:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)  # File gets your DEBUG
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    my_logger.addHandler(file_handler)  # Attach to your logger, not root
