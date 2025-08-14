"""rainfalltools — Tools for rainfall data processing and erosivity analysis."""
__version__ = "0.1.0"

# Re-export commonly used functions from submodules (wildcard for simplicity).
from .identify_storms import *  # noqa: F401,F403
from .filter_high_quality_sites import *  # noqa: F401,F403
from .process_intervals import *  # noqa: F401,F403
from .separate_storm_events import *  # noqa: F401,F403
from ..erosive_storms import *  # noqa: F401,F403
from .rainfall_erosivity import *  # noqa: F401,F403

__all__ = []  # You can list explicit public symbols here later for a tighter API.
