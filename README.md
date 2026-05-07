# counterstrike1k

[![PyPI](https://img.shields.io/pypi/v/counterstrike1k)](https://pypi.org/project/counterstrike1k/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

Small decoding helpers for the [**CounterStrike-1K**](https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K) dataset — a synchronized 10-POV Counter-Strike 2 dataset for video world modeling.

The dataset, paper, and evals live at **[github.com/AnirudhhRamesh/CounterStrike-1K](https://github.com/AnirudhhRamesh/CounterStrike-1K)**. This repository ships only the Python loader.

## Install

```bash
uv add datasets counterstrike1k
```

<details>
<summary>Using pip instead</summary>

```bash
pip install datasets counterstrike1k
```
</details>

## Use

```python
from datasets import load_dataset
from counterstrike1k import decode_sample

shards = load_dataset(
    "ArnieRamesh/CounterStrike-1K-360-wds", split="train", streaming=True,
)
sample = decode_sample(next(iter(shards)))

print(sample["actions"].shape)   # (frames,) tick, delta_pitch, delta_yaw, buttons bitmask
print(sample["state"].shape)     # (frames,) pos, view, weapon, ammo, hp, money, score, …
print(len(sample["video"]))      # mp4 bytes with synchronized audio
```

Tiny offline preview (one match-map, ~2 GB):

```python
from counterstrike1k import load_sample
for sample in load_sample():
    print(sample["metadata"]["sample_key"])
    break
```

Verify decoded actions actually align with the video — `overlay_frame` draws a HUD with WASD/FIRE/JUMP, mouse delta, HP/armor/money, and score onto any frame:

```python
from counterstrike1k import overlay_frame, overlay_video

overlay_frame(sample, 60)                          # PIL.Image, ready for display()
overlay_video(sample, "debug.mp4", max_frames=192) # full debug clip with audio
```

## API

| Symbol | What it does |
|---|---|
| `decode_sample(d)` | Turn one streamed WebDataset sample dict into numpy arrays + mp4 bytes + metadata |
| `decode_actions(b)` / `decode_state(b)` | Decode raw `.actions.bin` / `.state.bin` payloads |
| `unpack_buttons(actions)` | Expand the 12-button bitmask into one boolean array per button |
| `player_alive_mask(meta, state)` | Per-frame alive mask from metadata tick bounds |
| `load_sample(repo_id=...)` | Iterate the small preview repo as decoded sample dicts |
| `overlay_frame(sample, idx)` | Return a `PIL.Image` with the action/state HUD drawn on the video frame |
| `overlay_video(sample, path)` | Write a debug-overlay mp4 (preserves source audio) |
| `BUTTONS`, `ACTIONS_DTYPE`, `STATE_DTYPE` | Schema constants |

## Citation

```bibtex
@dataset{counterstrike1k2026,
  title     = {CounterStrike-1K: Synchronized Multi-POV Counter-Strike 2 for World Modeling},
  author    = {Ramesh, Anirudhh},
  year      = {2026},
  publisher = {Hugging Face},
  version   = {1.0.0},
  url       = {https://huggingface.co/datasets/ArnieRamesh/CounterStrike-1K}
}
```

## License

CC BY-NC 4.0 to the extent of the authors' rights. See [LICENSE](LICENSE). Counter-Strike 2 and underlying game assets remain property of Valve Corporation.
