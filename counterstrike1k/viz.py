"""Lightweight debug overlay for CounterStrike-1K samples.

Use this to verify that decoded actions/state align with the rendered video.
Pure PIL — no FFmpeg required for single-frame inspection. Video export uses
PyAV (an `[notebooks]` extra), so it stays optional for the core install.

```python
from counterstrike1k import load_sample, overlay_frame, overlay_video

sample = next(iter(load_sample()))
overlay_frame(sample, 60)              # PIL.Image, ready for display()
overlay_video(sample, "debug.mp4")     # full-clip mp4 with overlay HUD
```

Visual style: black-first, sharp corners, JetBrains Mono, team-colored accents.
Matches the media_claude release videos.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from counterstrike1k.schema import BUTTONS


# --- Design tokens (mirrors media_claude/design.py) -------------------------

PANEL_FILL_SOLID = (0, 0, 0, 188)      # stats / score: stay readable
PANEL_FILL_GHOST = (0, 0, 0, 128)      # inputs / mouse delta: 50% see-through
PANEL_LINE = (255, 255, 255, 56)       # hairline border
DIVIDER = (255, 255, 255, 28)
TEXT = (245, 248, 252, 255)
DIM = (170, 178, 192, 255)
MUTED = (108, 116, 128, 255)
CT = (102, 178, 255, 255)               # CT blue
T = (232, 180, 60, 255)                 # T yellow
OK = (110, 220, 140, 255)
LOW = (255, 100, 100, 255)
KEY_OFF_OUTLINE = (120, 130, 144, 200)
KEY_OFF_TEXT = (235, 240, 248, 235)
KEY_ON_TEXT = (10, 14, 18, 255)


_ASSETS = Path(__file__).resolve().parent / "assets" / "fonts"


@lru_cache(maxsize=64)
def _font(size: int, weight: str = "regular"):
    """JetBrains Mono at the requested weight; pure-Pillow fallback if missing."""

    from PIL import ImageFont

    candidates = {
        "regular": ["JetBrainsMono-Regular.ttf", "JetBrainsMono-Medium.ttf", "JetBrainsMono-Bold.ttf"],
        "medium":  ["JetBrainsMono-Medium.ttf", "JetBrainsMono-Bold.ttf", "JetBrainsMono-Regular.ttf"],
        "bold":    ["JetBrainsMono-Bold.ttf", "JetBrainsMono-Medium.ttf", "JetBrainsMono-Regular.ttf"],
    }.get(weight, ["JetBrainsMono-Regular.ttf"])
    for name in candidates:
        path = _ASSETS / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def team_accent(team_side: str | None) -> tuple[int, int, int, int]:
    if str(team_side or "").upper() == "T":
        return T
    return CT


# --- Shape primitives -------------------------------------------------------

def _rect(draw, box, *, fill=None, outline=None, width: int = 1) -> None:
    draw.rectangle(list(box), fill=fill, outline=outline, width=width)


def _hairline(draw, x0: int, y0: int, x1: int, y1: int, color=DIVIDER) -> None:
    draw.line([(x0, y0), (x1, y1)], fill=color, width=1)


def _panel(draw, box, *, ghost: bool = False) -> None:
    fill = PANEL_FILL_GHOST if ghost else PANEL_FILL_SOLID
    _rect(draw, box, fill=fill, outline=PANEL_LINE, width=1)


def _left_accent_bar(draw, box, color, *, width_px: int = 3) -> None:
    x0, y0, x1, y1 = box
    _rect(draw, (x0, y0, x0 + width_px, y1), fill=color)


def _top_accent_bar(draw, box, color, *, height_px: int = 2) -> None:
    x0, y0, x1, _ = box
    _rect(draw, (x0, y0, x1, y0 + height_px), fill=color)


def _text_size(draw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_key(draw, box, label: str, pressed: bool, accent, font) -> None:
    if pressed:
        _rect(draw, box, fill=accent, outline=accent, width=1)
        fg = KEY_ON_TEXT
    else:
        # No fill — panel's see-through background carries through.
        _rect(draw, box, fill=None, outline=KEY_OFF_OUTLINE, width=1)
        fg = KEY_OFF_TEXT
    x0, y0, x1, y1 = box
    tw, th = _text_size(draw, label, font)
    draw.text((x0 + ((x1 - x0) - tw) // 2, y0 + ((y1 - y0) - th) // 2 - 1), label, font=font, fill=fg)


# --- Frame decode helpers ---------------------------------------------------

def _decode_frames(video_bytes: bytes, indices: list[int]):
    import av

    wanted = set(indices)
    out: dict[int, Any] = {}
    with av.open(io.BytesIO(video_bytes)) as container:
        max_wanted = max(wanted)
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                out[i] = frame.to_image().convert("RGB")
                if len(out) == len(wanted):
                    break
            if i > max_wanted:
                break
    return out




# --- Public API -------------------------------------------------------------

def overlay_frame(sample: dict, frame_idx: int):
    """Return a PIL.Image with the action/state HUD overlaid on the video frame."""

    actions = sample["actions"]
    state = sample["state"]
    n = min(len(actions), len(state))
    if n == 0:
        raise ValueError("sample has zero frames")
    frame_idx = max(0, min(int(frame_idx), n - 1))

    frame = _decode_frames(sample["video"], [frame_idx]).get(frame_idx)
    if frame is None:
        raise ValueError(f"could not decode frame {frame_idx}")

    return _compose_overlay(frame, actions, state, frame_idx, sample.get("metadata", {}))


def _compose_overlay(frame, actions, state, frame_idx: int, metadata: dict):
    from PIL import ImageDraw

    img = frame.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    a = actions[frame_idx]
    s = state[frame_idx]
    accent = team_accent(metadata.get("team_side"))

    bits = int(a["buttons"])
    pressed = {name: bool((bits >> i) & 1) for i, name in enumerate(BUTTONS)}

    pad = 12
    _draw_top_left_stats(draw, pad, pad, s, accent)
    _draw_top_right_meta(draw, W - pad, pad, s, a, frame_idx, len(actions), metadata, accent)
    _draw_bottom_left_inputs(draw, pad, H - pad, pressed, accent)
    _draw_bottom_right_mouse(draw, W - pad, H - pad, float(a["delta_pitch"]), float(a["delta_yaw"]), accent)

    return img.convert("RGB")


# --- Top-left: HP / Armor (+H) / $ -----------------------------------------

def _format_money(balance: int) -> str:
    if balance == 65535:
        return "$—"
    return f"${int(balance)}"


def _format_int(value: int, sentinel: int = 255) -> str:
    if value == sentinel:
        return "—"
    return str(int(value))


def _draw_top_left_stats(draw, x: int, y: int, s, accent) -> None:
    f_val = _font(20, "bold")
    f_lab = _font(10, "medium")

    hp = int(s["health"])
    armor = int(s["armor_value"])
    helmet = bool(s["has_helmet"])
    money = int(s["balance"])

    chips: list[tuple[str, str, tuple[int, int, int, int]]] = [
        ("HP", _format_int(hp), OK if hp >= 50 else (LOW if hp <= 30 else TEXT)),
        ("ARMOR", _format_int(armor), TEXT if armor > 0 else MUTED),
    ]
    if helmet:
        chips.append(("HELM", "●", TEXT))
    chips.append(("MONEY", _format_money(money), TEXT))

    pad_x = 14
    chip_h = 50
    sized: list[tuple[str, str, tuple[int, int, int, int], int]] = []
    total_w = 0
    for lab, val, color in chips:
        vw, _ = _text_size(draw, val, f_val)
        lw, _ = _text_size(draw, lab, f_lab)
        cw = max(vw, lw) + 2 * pad_x
        sized.append((lab, val, color, cw))
        total_w += cw

    box = (x, y, x + total_w, y + chip_h)
    _panel(draw, box)
    _left_accent_bar(draw, box, accent, width_px=3)

    cx = x
    for i, (lab, val, color, cw) in enumerate(sized):
        if i > 0:
            _hairline(draw, cx, y + 8, cx, y + chip_h - 8)
        vw, vh = _text_size(draw, val, f_val)
        draw.text((cx + (cw - vw) // 2, y + 6), val, font=f_val, fill=color)
        lw, _ = _text_size(draw, lab, f_lab)
        draw.text((cx + (cw - lw) // 2, y + 6 + vh + 4), lab, font=f_lab, fill=DIM)
        cx += cw


# --- Top-right: T : CT, pov, round, frame, tick ----------------------------

def _draw_top_right_meta(draw, x_right: int, y: int, s, a, frame_idx: int, n_frames: int, metadata: dict, accent) -> None:
    f_score = _font(22, "bold")
    f_label = _font(10, "medium")
    f_meta = _font(11)
    f_dim = _font(10)

    t_score = _format_int(int(s["t_score"]))
    ct_score = _format_int(int(s["ct_score"]))
    pov_idx = metadata.get("pov_idx")
    round_idx = metadata.get("round_idx")
    team_side = str(metadata.get("team_side") or "").upper()

    t_label_w, t_label_h = _text_size(draw, "T", f_label)
    ct_label_w, _ = _text_size(draw, "CT", f_label)
    score_text = f"{t_score} · {ct_score}"
    score_w, score_h = _text_size(draw, score_text, f_score)

    pad_x = 14
    pad_y = 8
    score_row_w = t_label_w + 8 + score_w + 8 + ct_label_w + 2 * pad_x
    meta_lines = []
    if pov_idx is not None or team_side:
        side = (" " + team_side) if team_side else ""
        meta_lines.append(f"pov {int(pov_idx)}{side}" if pov_idx is not None else side.strip())
    if round_idx is not None:
        meta_lines.append(f"round {int(round_idx):03d}")
    meta_lines.append(f"frame {frame_idx}/{n_frames - 1}")
    meta_lines.append(f"tick {int(a['tick'])}")

    line_h = _text_size(draw, "tick 0", f_meta)[1] + 6
    meta_w = max(_text_size(draw, line, f_meta)[0] for line in meta_lines)
    panel_w = max(score_row_w, meta_w + 2 * pad_x)
    panel_h = score_h + 2 * pad_y + 6 + line_h * len(meta_lines) + pad_y

    panel_x = x_right - panel_w
    box = (panel_x, y, panel_x + panel_w, y + panel_h)
    _panel(draw, box)
    _top_accent_bar(draw, box, accent, height_px=2)

    cy = y + pad_y + 4
    cx = panel_x + (panel_w - score_row_w) // 2 + pad_x
    draw.text((cx, cy + (score_h - t_label_h) // 2 + 1), "T", font=f_label, fill=T)
    cx += t_label_w + 8
    draw.text((cx, cy), score_text, font=f_score, fill=TEXT)
    cx += score_w + 8
    draw.text((cx, cy + (score_h - t_label_h) // 2 + 1), "CT", font=f_label, fill=CT)

    cy += score_h + 8
    for line in meta_lines:
        draw.text((panel_x + pad_x, cy), line, font=f_meta, fill=DIM)
        cy += line_h


# --- Bottom-left: keyboard --------------------------------------------------

def _draw_bottom_left_inputs(draw, x: int, y_bottom: int, pressed: dict, accent) -> None:
    """Real-keyboard layout: Shift on the ASD row, Ctrl on the Space row, ERF on ASD."""

    f_key = _font(11, "bold")
    f_mod = _font(10, "bold")

    key_w = 32
    key_h = 28
    mod_w = 56
    gap = 4

    panel_pad_x = 10
    panel_pad_y = 8
    panel_x = x
    base_x = panel_x + panel_pad_x

    col_mod = base_x
    col_a = col_mod + mod_w + gap
    col_s = col_a + key_w + gap
    col_d = col_s + key_w + gap
    col_e = col_d + key_w + gap
    col_r = col_e + key_w + gap
    col_f = col_r + key_w + gap
    panel_w = (col_f + key_w) - panel_x + panel_pad_x

    panel_h = key_h * 3 + gap * 2 + 2 * panel_pad_y
    panel_y = y_bottom - panel_h

    box = (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h)
    _panel(draw, box, ghost=True)

    cy_top = panel_y + panel_pad_y
    cy_mid = cy_top + key_h + gap
    cy_bot = cy_mid + key_h + gap

    # Row 1 — W centered above S
    _draw_key(draw, (col_s, cy_top, col_s + key_w, cy_top + key_h),
              "W", pressed.get("FORWARD", False), accent, f_key)

    # Row 2 — Shift, A, S, D, E, R, F
    _draw_key(draw, (col_mod, cy_mid, col_mod + mod_w, cy_mid + key_h),
              "SHIFT", pressed.get("WALK", False), accent, f_mod)
    for col, label, btn in [
        (col_a, "A", "LEFT"),
        (col_s, "S", "BACK"),
        (col_d, "D", "RIGHT"),
        (col_e, "E", "USE"),
        (col_r, "R", "RELOAD"),
        (col_f, "F", "INSPECT"),
    ]:
        _draw_key(draw, (col, cy_mid, col + key_w, cy_mid + key_h),
                  label, pressed.get(btn, False), accent, f_key)

    # Row 3 — Ctrl + Space (spans A→D)
    _draw_key(draw, (col_mod, cy_bot, col_mod + mod_w, cy_bot + key_h),
              "CTRL", pressed.get("DUCK", False), accent, f_mod)
    space_x1 = col_d + key_w
    _draw_key(draw, (col_a, cy_bot, space_x1, cy_bot + key_h),
              "SPACE", pressed.get("JUMP", False), accent, f_mod)

    # Mouse panel — separate, vertical, right of the keyboard panel.
    mouse_w = 60
    mouse_h = key_h * 2 + gap + 2 * panel_pad_y
    mouse_x = panel_x + panel_w + 8
    mouse_y = panel_y + (panel_h - mouse_h)
    mbox = (mouse_x, mouse_y, mouse_x + mouse_w, mouse_y + mouse_h)
    _panel(draw, mbox, ghost=True)
    my = mouse_y + panel_pad_y
    for label, btn in [("LMB", "FIRE"), ("RMB", "RIGHTCLICK")]:
        _draw_key(draw, (mouse_x + 8, my, mouse_x + mouse_w - 8, my + key_h),
                  label, pressed.get(btn, False), accent, f_mod)
        my += key_h + gap


# --- Bottom-right: mouse delta widget ---------------------------------------

def _draw_bottom_right_mouse(draw, x_right: int, y_bottom: int, dpitch: float, dyaw: float, accent) -> None:
    f_label = _font(9, "medium")
    f_val = _font(10)

    box_size = 116
    panel_x = x_right - box_size
    panel_y = y_bottom - box_size
    box = (panel_x, panel_y, panel_x + box_size, panel_y + box_size)
    _panel(draw, box, ghost=True)
    _top_accent_bar(draw, box, accent, height_px=2)

    cx = panel_x + box_size / 2
    cy = panel_y + box_size / 2 + 6
    draw.line([cx - 26, cy, cx + 26, cy], fill=(118, 128, 142, 130), width=1)
    draw.line([cx, cy - 26, cx, cy + 26], fill=(118, 128, 142, 130), width=1)

    mag = max(abs(dyaw), abs(dpitch))
    scale = 26.0 / max(0.001, mag if mag > 1 else 1)
    ax = cx + dyaw * scale
    ay = cy - dpitch * scale  # invert: pitch up = arrow up
    draw.line([cx, cy, ax, ay], fill=accent, width=3)
    draw.ellipse([ax - 3, ay - 3, ax + 3, ay + 3], fill=accent)

    draw.text((panel_x + 10, panel_y + 6), "Δ MOUSE", font=f_label, fill=DIM)
    draw.text(
        (panel_x + 10, panel_y + box_size - 16),
        f"{dyaw:+5.1f} / {dpitch:+5.1f}",
        font=f_val,
        fill=DIM,
    )


def overlay_video(
    sample: dict,
    out_path: str | Path,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
) -> Path:
    """Write a debug-overlay mp4 of the sample to `out_path`.

    Audio from the source mp4 is preserved (stream-copied, no re-encode), so
    you keep the original synchronized game audio without quality loss.

    Requires `pyav` (`pip install av`). Returns the output path.
    """

    import av
    import numpy as np
    from fractions import Fraction

    actions = sample["actions"]
    state = sample["state"]
    n = min(len(actions), len(state))
    if n == 0:
        raise ValueError("sample has zero frames")
    if max_frames is not None:
        n = min(n, int(max_frames))

    fps_actual = float(fps) if fps else float(sample.get("metadata", {}).get("fps") or 32)
    duration_s = n / fps_actual
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inp = av.open(io.BytesIO(sample["video"]))
    try:
        video_in = inp.streams.video[0]
        audio_in = next(iter(inp.streams.audio), None)

        out = av.open(str(out_path), mode="w")
        try:
            rate = Fraction(fps_actual).limit_denominator(1000)
            video_out = out.add_stream("h264", rate=rate)
            video_out.width = video_in.codec_context.width
            video_out.height = video_in.codec_context.height
            video_out.pix_fmt = "yuv420p"
            video_out.options = {"crf": "20"}

            audio_out = out.add_stream_from_template(audio_in) if audio_in is not None else None

            frame_idx = 0
            for packet in inp.demux(video_in, audio_in) if audio_in is not None else inp.demux(video_in):
                if packet.dts is None:
                    continue

                if packet.stream is video_in:
                    if frame_idx >= n:
                        continue
                    for frame in packet.decode():
                        if frame_idx >= n:
                            break
                        pil = frame.to_image().convert("RGB")
                        composed = _compose_overlay(
                            pil, actions, state, frame_idx, sample.get("metadata", {}),
                        )
                        arr = np.asarray(composed)
                        vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
                        for p in video_out.encode(vf):
                            out.mux(p)
                        frame_idx += 1
                elif audio_out is not None and packet.stream is audio_in:
                    # Truncate audio to match the (possibly shortened) video.
                    if packet.pts is not None:
                        pts_s = float(packet.pts * audio_in.time_base)
                        if pts_s > duration_s:
                            continue
                    packet.stream = audio_out
                    out.mux(packet)

            for p in video_out.encode():
                out.mux(p)
        finally:
            out.close()
    finally:
        inp.close()

    return out_path
