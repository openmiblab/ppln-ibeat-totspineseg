from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ibeat-totspineseg")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"


from . import (
    stage_01_auto_segment,
    stage_02_manual_segment,
    stage_03_display,
    stage_04_measure
)
