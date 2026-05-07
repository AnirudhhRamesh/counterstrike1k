"""Tests for the public counterstrike1k decoding helpers."""

from __future__ import annotations

import json

import numpy as np

from counterstrike1k import (
    ACTIONS_DTYPE,
    BUTTONS,
    STATE_DTYPE,
    decode_actions,
    decode_sample,
    decode_state,
    player_alive_mask,
    unpack_buttons,
)


def _fake_actions(frames: int = 4, start_tick: int = 100) -> np.ndarray:
    actions = np.zeros(frames, dtype=ACTIONS_DTYPE)
    actions["tick"] = np.arange(start_tick, start_tick + 2 * frames, 2, dtype=np.uint32)
    actions["delta_pitch"] = np.linspace(-0.5, 0.5, frames, dtype=np.float32)
    actions["delta_yaw"] = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    actions["buttons"] = 0b0000_0000_1000_0001  # FORWARD + FIRE
    return actions


def _fake_state(frames: int = 4, start_tick: int = 100) -> np.ndarray:
    state = np.zeros(frames, dtype=STATE_DTYPE)
    state["tick"] = np.arange(start_tick, start_tick + 2 * frames, 2, dtype=np.uint32)
    state["health"] = 100
    state["t_score"] = 1
    state["ct_score"] = 2
    return state


def _fake_metadata(sample_key: str = "match_abc123__r000__p00", frames: int = 4):
    return {
        "sample_key": sample_key,
        "match_id": "abc123",
        "round_idx": 0,
        "round_id": "match_abc123__r000",
        "pov_idx": 0,
        "map_slug": "dust2",
        "frames": frames,
        "fps": 32.0,
        "alive_start_tick": 100,
        "alive_end_tick": 104,
    }


def test_decode_actions_roundtrip():
    actions = _fake_actions(frames=4)
    decoded = decode_actions(actions.tobytes())
    assert decoded.dtype == ACTIONS_DTYPE
    assert decoded.shape == (4,)
    assert decoded["tick"].tolist() == [100, 102, 104, 106]


def test_decode_state_roundtrip():
    state = _fake_state(frames=4)
    decoded = decode_state(state.tobytes())
    assert decoded.dtype == STATE_DTYPE
    assert decoded["health"].tolist() == [100, 100, 100, 100]


def test_unpack_buttons_returns_one_array_per_button():
    actions = _fake_actions(frames=4)
    buttons = unpack_buttons(actions)
    assert set(buttons.keys()) == set(BUTTONS)
    assert buttons["FORWARD"].tolist() == [True, True, True, True]
    assert buttons["FIRE"].tolist() == [True, True, True, True]
    assert buttons["JUMP"].tolist() == [False, False, False, False]


def test_decode_sample_handles_bytes_inputs():
    metadata = _fake_metadata()
    sample = decode_sample({
        "__key__": metadata["sample_key"],
        "mp4": b"FAKEMP4",
        "actions.bin": _fake_actions().tobytes(),
        "state.bin": _fake_state().tobytes(),
        "events.json": json.dumps({"events": [{"tick": 100, "frame_idx": 0, "type": "round_freeze_end"}]}).encode(),
        "json": json.dumps(metadata).encode(),
    })

    assert sample["key"] == metadata["sample_key"]
    assert sample["actions"].dtype == ACTIONS_DTYPE
    assert sample["state"].dtype == STATE_DTYPE
    assert sample["video"] == b"FAKEMP4"
    assert isinstance(sample["events"], list)
    assert sample["events"][0]["type"] == "round_freeze_end"
    assert sample["metadata"]["match_id"] == "abc123"


def test_decode_sample_accepts_already_decoded_json():
    """datasets.load_dataset on WebDataset may auto-decode .json payloads."""

    metadata = _fake_metadata()
    sample = decode_sample({
        "__key__": metadata["sample_key"],
        "mp4": b"FAKEMP4",
        "actions.bin": _fake_actions().tobytes(),
        "state.bin": _fake_state().tobytes(),
        "events.json": {"events": []},
        "json": metadata,
    })

    assert sample["metadata"]["sample_key"] == metadata["sample_key"]
    assert sample["events"] == []


def test_decode_sample_accepts_video_decode_false_payload():
    """datasets.Video(decode=False) wraps MP4 bytes in a dict."""

    metadata = _fake_metadata()
    sample = decode_sample({
        "__key__": metadata["sample_key"],
        "mp4": {"bytes": b"FAKEMP4", "path": None},
        "actions.bin": _fake_actions().tobytes(),
        "state.bin": _fake_state().tobytes(),
        "events.json": {"events": []},
        "json": metadata,
    })

    assert sample["video"] == b"FAKEMP4"
    assert sample["actions"].shape == (4,)


def test_player_alive_mask_uses_metadata_window():
    state = _fake_state(frames=4)
    metadata = _fake_metadata(frames=4)
    mask = player_alive_mask(metadata, state)
    # alive_end_tick == 104 is exclusive, so frames at ticks 100, 102 are alive.
    assert mask.tolist() == [True, True, False, False]
