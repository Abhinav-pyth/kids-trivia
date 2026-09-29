"""
Kids Trivia Video Builder -- core backend engine
================================================

Generates a 9:16 vertical (1080x1920) trivia video for YouTube Shorts / mobile.

Pipeline
--------
    trivia dict (JSON)  ->  KidsTriviaVideoBuilder.build_clip()  ->  CompositeVideoClip
                          KidsTriviaVideoBuilder.render()        ->  web-optimized H.264 .mp4

Timeline of the generated video (10 s @ 30 fps)
-----------------------------------------------
    0 s .......... 5 s   COUNTDOWN PHASE
        * question + options already on screen (they stay visible all 10 s)
        * neon progress bar at the bottom shrinks smoothly from 100% to 0% width
        * big "5 4 3 2 1" digits above the bar
    5 s ......... 10 s   REVEAL PHASE
        * bar + digits disappear (their duration simply ends at 5 s)
        * correct option turns NEON GREEN (+ glowing card + check badge)
        * wrong options fade back to ``dimmed_opacity`` and stay white
        * optional "TIME'S UP!" banner pops in

Coordinate system (READ THIS FIRST TO ADJUST THE LAYOUT)
-------------------------------------------------------
MoviePy composites every layer with a TOP-LEFT origin:

    x grows to the RIGHT : 0 ......... VIDEO_WIDTH   (1080 px)
    y grows DOWNWARD     : 0 ......... VIDEO_HEIGHT  (1920 px)

``clip.with_position((x, y))`` puts the TOP-LEFT CORNER of that clip at (x, y).
Formulas used throughout this file (W/H = canvas, w/h = clip size):

    horizontal centering        x = (W - w) // 2
    vertical centering          y = (H - h) // 2
    anchor by BOTTOM edge       y = y_bottom - h
    percentage placement        x = int(W * pct_x),  y = int(H * pct_y)

Every position here is expressed as a PERCENTAGE of the canvas (``LayoutConfig``)
so the design scales automatically if you change ``VideoConfig.width/height``
(e.g. render 540x960 drafts ~4x faster, then re-render full size).

Layer order (``layer_index``, higher is drawn on top):
    0 background | 1 brand pill | 2 question card | 3 option cards
    4 countdown bar | 5 countdown digits | 6 reveal overlays | 7 footer/banner

Requirements
------------
    Python >= 3.9, moviepy >= 2.x (pip install moviepy), ffmpeg on PATH.
    Text/shapes are rendered with Pillow + NumPy, so ImageMagick is NOT needed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from moviepy import (
    AudioClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoClip,
)

logger = logging.getLogger("kids_trivia")

RGB = Tuple[int, int, int]
Box = Tuple[int, int, int, int]          # (x, y, width, height)


# --------------------------------------------------------------------------- #
# Colors
# --------------------------------------------------------------------------- #

def hex_to_rgb(value: str) -> RGB:
    """'#3EC1D3' -> (62, 193, 211). Accepts '#rrggbb' or 'rrggbb'."""
    value = value.lstrip("#")
    if len(value) != 6 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"Invalid hex color: {value!r} (expected '#rrggbb')")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


NEON_GREEN: RGB = (57, 255, 20)         # highlight colour of the revealed answer


# --------------------------------------------------------------------------- #
# Input data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TriviaQuestion:
    """Validated input payload.

    Expected JSON/dict shape::

        {
          "question": "Which animal is the tallest in the world?",
          "options": ["A) Elephant", "B) Giraffe", "C) Blue Whale"],
          "correct_answer": "B) Giraffe"
        }
    """

    question: str
    options: List[str]
    correct_answer: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriviaQuestion":
        """Build + validate a question from a plain dict (or parsed JSON)."""
        if not isinstance(data, dict):
            raise TypeError("Trivia payload must be a dict / JSON object")
        try:
            question = str(data["question"]).strip()
            options = [str(opt).strip() for opt in data["options"]]
            correct_answer = str(data["correct_answer"]).strip()
        except KeyError as exc:                        # missing required key
            raise ValueError(f"Trivia payload is missing the key {exc}") from exc
        except TypeError as exc:                       # 'options' not iterable
            raise ValueError("'options' must be a list of strings") from exc

        if not question:
            raise ValueError("'question' must be a non-empty string")
        if len(options) < 2:
            raise ValueError("'options' needs at least 2 entries")
        if any(not opt for opt in options):
            raise ValueError("'options' contains an empty entry")
        if correct_answer not in options:
            raise ValueError(
                f"'correct_answer' ({correct_answer!r}) must exactly match one of the options"
            )
        return cls(question=question, options=options, correct_answer=correct_answer)

    @property
    def correct_index(self) -> int:
        """Index of the winning option -> decides which overlay turns green."""
        return self.options.index(self.correct_answer)


# --------------------------------------------------------------------------- #
# Layout configuration: ALL positions are fractions of the canvas
# --------------------------------------------------------------------------- #

@dataclass
class LayoutConfig:
    """Vertical rhythm of the 9:16 frame, from the top (0.0) to the bottom (1.0).

    Tweak these numbers only -- the builder converts them to pixels for you.
    """

    side_margin_pct: float = 0.075      # left/right safe area (mobile UI overlays!)
    brand_top_pct: float = 0.020        # "KIDS TRIVIA" pill: TOP edge
    question_top_pct: float = 0.075     # question card: TOP edge
    question_height_pct: float = 0.165  # question card: HEIGHT
    options_center_pct: float = 0.415   # option stack: vertical CENTER
    option_gap_pct: float = 0.020       # gap between two option cards
    bar_top_pct: float = 0.775          # progress-bar track: TOP edge
    bar_width_pct: float = 0.86         # progress bar: FULL (100%) WIDTH
    bar_height_pct: float = 0.028       # progress bar: THICKNESS
    counter_center_pct: float = 0.870   # countdown digits: vertical CENTER
    timesup_center_pct: float = 0.675   # "TIME'S UP!" banner: vertical CENTER
    cta_bottom_pct: float = 0.965       # footer CTA: BOTTOM edge


@dataclass
class VideoConfig:
    """Everything the renderer needs. The defaults satisfy the product spec."""

    # --- canvas / timing ----------------------------------------------------
    width: int = 1080                   # X axis: 0 (left)  -> 1080 (right)
    height: int = 1920                  # Y axis: 0 (top)   -> 1920 (bottom)
    fps: int = 30                       # YouTube Shorts friendly frame rate
    duration: float = 10.0              # total length in seconds
    reveal_time: float = 5.0            # end of countdown / start of the reveal
    background_color: RGB = field(default_factory=lambda: hex_to_rgb("#B9A7E4"))

    # --- typography ---------------------------------------------------------
    font_path: str = ""                 # "" -> auto-detect a bold TTF (see below)
    question_font_size: int = 72
    option_font_size: int = 64
    countdown_font_size: int = 110
    brand_font_size: int = 40
    text_color: RGB = (255, 255, 255)
    stroke_color: RGB = field(default_factory=lambda: hex_to_rgb("#3B2A63"))
    stroke_width: int = 3               # dark outline => readable on pastel colours
    interline: int = 10                 # extra line spacing inside wrapped text
    dimmed_opacity: float = 0.45        # final opacity of the WRONG answers
    reveal_fade_in: float = 0.35        # seconds spent on the reveal transition

    # --- shapes / decorations ----------------------------------------------
    card_radius: int = 34               # corner radius of every card
    card_body_alpha: int = 150          # opacity (0-255) of the card bodies
    show_check_badge: bool = True       # green circle + white tick on the answer
    show_times_up_banner: bool = True   # "TIME'S UP!" pop-in during the reveal
    audio_enabled: bool = True          # synthesized tick/ding cues (no assets)

    # --- export (web optimized) --------------------------------------------
    codec: str = "libx264"              # H.264
    preset: str = "medium"              # encoding speed vs. compression
    crf: int = 21                       # constant rate factor (18=near-lossless)
    maxrate: str = "4000k"              # caps the bitrate for fast web delivery
    bufsize: str = "8M"
    pix_fmt: str = "yuv420p"            # REQUIRED for broad browser/mobile playback
    profile: str = "high"
    threads: int = 4

    layout: LayoutConfig = field(default_factory=LayoutConfig)

    # ---------------------------------------------------------------- derived
    @property
    def px(self) -> Dict[str, int]:
        """Percentage layout -> absolute pixel coordinates (top-left origin)."""
        W, H, L = self.width, self.height, self.layout
        return {
            "side_margin": int(W * L.side_margin_pct),
            "text_max_width": int(W * (1.0 - 2.0 * L.side_margin_pct)),
            "brand_top": int(H * L.brand_top_pct),
            "question_top": int(H * L.question_top_pct),
            "question_height": int(H * L.question_height_pct),
            "options_center": int(H * L.options_center_pct),
            "option_gap": int(H * L.option_gap_pct),
            "bar_top": int(H * L.bar_top_pct),
            "bar_full_width": int(W * L.bar_width_pct),
            "bar_height": max(12, int(H * L.bar_height_pct)),
            "counter_center": int(H * L.counter_center_pct),
            "timesup_center": int(H * L.timesup_center_pct),
            "cta_bottom": int(H * L.cta_bottom_pct),
        }

    @property
    def reveal_duration(self) -> float:
        """Length of the reveal phase (seconds of footage after the cut)."""
        return max(0.0, self.duration - self.reveal_time)

    def resolved_font_path(self) -> str:
        """Return a usable bold TTF path; fails fast with an actionable message."""
        if self.font_path:
            if os.path.isfile(self.font_path):
                return self.font_path
            raise FileNotFoundError(f"VideoConfig.font_path not found: {self.font_path}")
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",      # Linux
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",       # Linux alt
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",         # macOS
            "C:/Windows/Fonts/arialbd.ttf",                              # Windows
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise RuntimeError(
            "No bold TTF font found. Set VideoConfig.font_path to a real .ttf/.otf file."
        )


# --------------------------------------------------------------------------- #
# Low-level drawing helpers (Pillow -> MoviePy masks / clips)
# --------------------------------------------------------------------------- #

def _pil_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def _text_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    box = font.getbbox(text)
    return box[2] - box[0]


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """Greedy word-wrap so long questions never bleed outside the safe area."""
    out: List[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            trial = f"{current} {word}".strip()
            if current and _text_width(font, trial) > max_width:
                out.append(current)
                current = word
            else:
                current = trial
        out.append(current)
    return "\n".join(out)


def rounded_rect_image(size: Tuple[int, int], radius: int, fill: int = 255,
                       blur: int = 0) -> Image.Image:
    """Greyscale 'L' image holding a rounded rectangle -> reusable as a mask."""
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).rounded_rectangle(
        [0, 0, w - 1, h - 1], radius=max(0, min(radius, h // 2)), fill=int(fill))
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img


def mask_from_image(mask_img: Image.Image, duration: float) -> ImageClip:
    """Convert a greyscale Pillow image into a MoviePy mask clip (values 0..1)."""
    array = np.asarray(mask_img, dtype=np.float32) / 255.0
    return ImageClip(array, is_mask=True).with_duration(duration)


def pad_mask(mask_img: Image.Image, dx: int, dy: int, canvas: Tuple[int, int],
             duration: float) -> ImageClip:
    """Embed a smaller mask into a bigger transparent canvas at (dx, dy)."""
    cw, ch = max(1, int(canvas[0])), max(1, int(canvas[1]))
    padded = Image.new("L", (cw, ch), 0)
    padded.paste(mask_img, (int(dx), int(dy)))
    return mask_from_image(padded, duration)


def solid_mask_clip(wh: Tuple[int, int], value: float, duration: float) -> ImageClip:
    """Constant-opacity mask covering a whole clip (used for fades/dimming)."""
    array = np.full((max(1, int(wh[1])), max(1, int(wh[0]))), float(value),
                    dtype=np.float32)
    return ImageClip(array, is_mask=True).with_duration(duration)


def attach_animated_mask(clip: VideoClip, mask_fn: Callable[[float, np.ndarray], np.ndarray]
                         ) -> VideoClip:
    """Wrap ``clip`` so its alpha channel is post-processed by ``mask_fn``.

    ``mask_fn(t, mask_frame) -> new_mask_frame`` receives the ORIGINAL alpha of
    the clip (e.g. the glyph shape of a TextClip) and must return the same-size
    float array. This is how the reveal phase animates opacity: MoviePy v2 has
    no time-varying ``with_opacity``, but a frame function does exactly that.
    """
    base_get_frame = clip.get_frame
    base_mask = clip.mask
    if base_mask is None:                                  # opaque fallback
        base_mask = solid_mask_clip(clip.size, 1.0, clip.duration or 0.0)
    base_mask_frame = base_mask.get_frame

    def frame_function(t: float) -> np.ndarray:
        return base_get_frame(t)

    def mask_function(t: float) -> np.ndarray:
        return mask_fn(t, base_mask_frame(t - (clip.start or 0)))

    wrapped = VideoClip(frame_function, duration=clip.duration)
    wrapped.size = clip.size
    wrapped.pos = clip.pos                                 # keep placement
    animated = VideoClip(mask_function, is_mask=True, duration=clip.duration)
    animated.size = clip.size
    animated.pos = clip.pos
    wrapped.mask = animated
    wrapped.layer_index = clip.layer_index
    return wrapped


# --------------------------------------------------------------------------- #
# Audio factory (optional): tiny synthesized cues, zero external assets
# --------------------------------------------------------------------------- #

def _tone(freq: float, dur: float, sample_rate: int, volume: float = 0.2,
          decay: float = 8.0, harmonics: Sequence[float] = (1.0,)) -> np.ndarray:
    """Decaying sine 'beep' as mono float32 samples."""
    t = np.linspace(0.0, dur, int(round(dur * sample_rate)), endpoint=False)
    wave = np.zeros_like(t)
    for mult in harmonics:
        wave += np.sin(2 * np.pi * freq * mult * t) / len(harmonics)
    envelope = np.exp(-decay * t) * np.minimum(1.0, t / 0.006)   # soft attack
    return (volume * envelope * wave).astype(np.float32)


def build_audio_track(cfg: VideoConfig) -> Optional[AudioClip]:
    """One tick per countdown second + a rising two-note 'ding' at the reveal."""
    if not cfg.audio_enabled:
        return None
    sr = 44_100                                   # audio sample rate (not video fps)
    segments: List[np.ndarray] = [np.zeros(int(0.15 * sr), np.float32)]
    ticks = int(max(0.0, min(cfg.reveal_time, cfg.duration) - 1.0))
    for step in range(ticks):                     # 5 ... 2
        segments.append(_tone(880.0 + 55.0 * step, 0.09, sr, 0.14, 22.0, (1.0, 2.0)))
        segments.append(np.zeros(int(0.91 * sr), np.float32))
    if cfg.reveal_duration > 0:                   # celebratory ding right at 5 s
        lead = max(0.0, cfg.reveal_time - sum(s.size for s in segments) / sr)
        segments.append(np.zeros(int(round(lead * sr)), np.float32))
        segments.append(_tone(1046.5, 0.16, sr, 0.24, 9.0, (1.0, 1.5, 2.0)))
        segments.append(_tone(1568.0, 0.70, sr, 0.24, 4.0, (1.0, 1.5, 2.0)))
    mono = np.concatenate(segments)[: int(round(cfg.duration * sr))]
    if mono.size < cfg.duration * sr:             # pad the tail with silence
        mono = np.concatenate([mono, np.zeros(int(cfg.duration * sr) - mono.size,
                                             np.float32)])
    stereo = np.vstack([mono, mono]).astype(np.float32)
    n = stereo.shape[1]

    def frame_function(t: float) -> np.ndarray:
        index = np.clip((np.atleast_1d(t) * sr).astype(int), 0, n - 1)
        if index.size == 1:
            return stereo[:, int(index[0])]
        return stereo[:, index]

    return AudioClip(frame_function, duration=float(stereo.shape[1]) / sr, fps=sr)
