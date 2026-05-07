"""CounterStrike-1K: decoding helpers for the public CS2 multi-POV dataset.

The primary access path is `datasets.load_dataset(...)` against the public
WebDataset shards. This package only provides:

- numpy dtypes for `actions.bin` and `state.bin`
- `decode_sample(...)` to turn one streamed shard sample into numpy arrays
- `load_sample(...)` for offline iteration of the small preview repo
- `overlay_frame(...)` / `overlay_video(...)` to verify action/state alignment
"""

from counterstrike1k.loader import (
    METADATA_REPO,
    SAMPLE_PREVIEW_REPO,
    SHARDS_360_REPO,
    SHARDS_720_REPO,
    decode_sample,
    load_sample,
)
from counterstrike1k.schema import (
    ACTIONS_DTYPE,
    BUTTONS,
    STATE_DTYPE,
    decode_actions,
    decode_state,
    player_alive_mask,
    unpack_buttons,
)
from counterstrike1k.viz import overlay_frame, overlay_video

__all__ = [
    "ACTIONS_DTYPE",
    "BUTTONS",
    "METADATA_REPO",
    "SAMPLE_PREVIEW_REPO",
    "SHARDS_360_REPO",
    "SHARDS_720_REPO",
    "STATE_DTYPE",
    "decode_actions",
    "decode_sample",
    "decode_state",
    "load_sample",
    "overlay_frame",
    "overlay_video",
    "player_alive_mask",
    "unpack_buttons",
]
__version__ = "1.0.0"
