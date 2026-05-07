"""Numpy dtypes and decoders for CounterStrike-1K binary sidecars."""

from __future__ import annotations

import numpy as np


BUTTONS: list[str] = [
    "FORWARD",
    "BACK",
    "LEFT",
    "RIGHT",
    "JUMP",
    "DUCK",
    "WALK",
    "FIRE",
    "RIGHTCLICK",
    "RELOAD",
    "INSPECT",
    "USE",
]


ACTIONS_DTYPE = np.dtype([
    ("tick", "<u4"),
    ("delta_pitch", "<f4"),
    ("delta_yaw", "<f4"),
    ("buttons", "<u2"),
])
assert ACTIONS_DTYPE.itemsize == 14, ACTIONS_DTYPE.itemsize


STATE_DTYPE = np.dtype([
    ("tick", "<u4"),
    ("pitch", "<f4"),
    ("yaw", "<f4"),
    ("pos_x", "<f4"),
    ("pos_y", "<f4"),
    ("pos_z", "<f4"),
    ("active_weapon", "u1"),
    ("active_weapon_id", "u1"),
    ("ammo_clip", "u1"),
    ("ammo_reserve", "u1"),
    ("health", "u1"),
    ("armor_value", "u1"),
    ("balance", "<u2"),
    ("t_score", "u1"),
    ("ct_score", "u1"),
    ("has_helmet", "u1"),
    ("has_defuser", "u1"),
    ("has_bomb", "u1"),
])
assert STATE_DTYPE.itemsize == 37, STATE_DTYPE.itemsize


def decode_actions(data: bytes) -> np.ndarray:
    """Decode an `{sample_key}.actions.bin` payload into a structured array."""

    if len(data) % ACTIONS_DTYPE.itemsize != 0:
        raise ValueError(
            f"actions payload of {len(data)} bytes is not a multiple of {ACTIONS_DTYPE.itemsize}"
        )
    return np.frombuffer(data, dtype=ACTIONS_DTYPE)


def decode_state(data: bytes) -> np.ndarray:
    """Decode an `{sample_key}.state.bin` payload into a structured array."""

    if len(data) % STATE_DTYPE.itemsize != 0:
        raise ValueError(
            f"state payload of {len(data)} bytes is not a multiple of {STATE_DTYPE.itemsize}"
        )
    return np.frombuffer(data, dtype=STATE_DTYPE)


def unpack_buttons(actions: np.ndarray) -> dict[str, np.ndarray]:
    """Expand the uint16 button bitmask into one boolean array per button."""

    bits = actions["buttons"].astype(np.uint16)
    return {name: ((bits >> i) & 1).astype(bool) for i, name in enumerate(BUTTONS)}


def player_alive_mask(metadata: dict, state: np.ndarray) -> np.ndarray:
    """Frame-level player-alive mask derived from metadata tick bounds."""

    if len(state) == 0:
        return np.zeros((0,), dtype=np.bool_)
    ticks = state["tick"].astype(np.int64)
    start = int(metadata.get("alive_start_tick") or metadata.get("clip_start_tick") or 0)
    end = int(metadata.get("alive_end_tick") or metadata.get("clip_end_tick") or start)
    return (ticks >= start) & (ticks < end)
