"""Tiny convenience helpers around `datasets.load_dataset` and the preview repo.

Most users should reach for `datasets.load_dataset(...)` directly. These helpers
exist only to (1) decode a sample dict in one call and (2) iterate the small
preview repo without writing 20 lines of `huggingface_hub` glue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from counterstrike1k.schema import decode_actions, decode_state


SAMPLE_PREVIEW_REPO = "ArnieRamesh/CounterStrike-1K-sample"
SHARDS_360_REPO = "ArnieRamesh/CounterStrike-1K-360-wds"
SHARDS_720_REPO = "ArnieRamesh/CounterStrike-1K-720-wds"
METADATA_REPO = "ArnieRamesh/CounterStrike-1K"


def _maybe_loads(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return json.loads(value)
    if isinstance(value, str):
        return json.loads(value)
    return value


def _bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError(f"expected bytes for binary member, got {type(value).__name__}")


def decode_sample(sample: dict) -> dict:
    """Decode one CounterStrike-1K sample dict into numpy arrays + metadata.

    Accepts the dict produced by ``datasets.load_dataset(..., streaming=True)``
    over a CounterStrike-1K WebDataset shard. Each member key matches the file
    suffix (``mp4``, ``actions.bin``, ``state.bin``, ``events.json``, ``json``).
    """

    metadata = _maybe_loads(sample["json"])
    events_payload = _maybe_loads(sample["events.json"])
    events = events_payload.get("events", events_payload) if isinstance(events_payload, dict) else events_payload

    return {
        "key": str(sample.get("__key__") or metadata.get("sample_key", "")),
        "video": _bytes(sample["mp4"]),
        "actions": decode_actions(_bytes(sample["actions.bin"])),
        "state": decode_state(_bytes(sample["state.bin"])),
        "events": events,
        "metadata": metadata,
    }


def load_sample(repo_id: str = SAMPLE_PREVIEW_REPO) -> Iterator[dict]:
    """Iterate the small preview repo as decoded sample dicts.

    Downloads the repo on first call (a few hundred MB, one match-map of 10
    synchronized POV rounds) and yields decoded samples in manifest order.
    """

    from huggingface_hub import snapshot_download
    import pandas as pd

    root = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
    manifest = pd.read_parquet(root / "manifest.parquet")
    for _, row in manifest.iterrows():
        key = str(row["sample_key"])
        yield decode_sample({
            "__key__": key,
            "mp4": _read(root, ["videos/360p", "videos"], f"{key}.mp4"),
            "actions.bin": _read(root, ["actions"], f"{key}.actions.bin"),
            "state.bin": _read(root, ["state"], f"{key}.state.bin"),
            "events.json": _read(root, ["events"], f"{key}.events.json"),
            "json": _read(root, ["metadata"], f"{key}.json"),
        })


def _read(root: Path, candidates: list[str], filename: str) -> bytes:
    for sub in candidates:
        path = root / sub / filename
        if path.exists():
            return path.read_bytes()
    raise FileNotFoundError(f"could not find {filename} under {[str(root / c) for c in candidates]}")
