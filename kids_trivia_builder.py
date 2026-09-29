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
import math
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


# --------------------------------------------------------------------------- #
# The engine itself
# --------------------------------------------------------------------------- #

@dataclass
class RenderResult:
    """Small result object handed back to callers (CLI, tests, web API)."""

    output_path: str
    size_bytes: int
    duration: float
    width: int
    height: int
    fps: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "output_path": self.output_path,
            "size_bytes": self.size_bytes,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


class KidsTriviaVideoBuilder:
    """Turns one trivia dict into a ready-to-render ``CompositeVideoClip``.

    Public API
    ----------
        builder = KidsTriviaVideoBuilder(TriviaQuestion.from_dict(payload))
        clip    = builder.build_clip()                  # in-memory composition
        result  = builder.render("out.mp4")             # H.264 file on disk

    Static helpers (used by the FastAPI / Vercel serverless entry point)
    -------------------------------------------------------------------
        KidsTriviaVideoBuilder.from_payload(payload)    # dict -> builder
        KidsTriviaVideoBuilder.config_from_payload(p)   # dict -> VideoConfig
    """

    BRAND_LABEL = "KIDS TRIVIA"
    CTA_LABEL = "Subscribe for more!"

    def __init__(self, question: TriviaQuestion,
                 config: Optional[VideoConfig] = None) -> None:
        self.question = question
        self.config = config or VideoConfig()
        self._font_path = self.config.resolved_font_path()
        self._px = self.config.px                     # cached pixel geometry

    # ------------------------------------------------------------------ setup

    @staticmethod
    def config_from_payload(data: Dict[str, Any]) -> VideoConfig:
        """Build a ``VideoConfig`` from JSON so the layout is tunable per request.

        Only keys that exist on :class:`VideoConfig` / :class:`LayoutConfig` are
        honoured; unknown keys are ignored on purpose (forward compatible API).

            {"question": ..., "options": [...], "correct_answer": ...,
             "config": {"background_color": "#A8D8EA", "fps": 30,
                        "preset": "veryfast",
                        "layout": {"options_center_pct": 0.45}}}
        """
        cfg = VideoConfig()
        raw = data.get("config") or {}
        if not isinstance(raw, dict):
            raise ValueError("'config' must be a JSON object")
        layout_raw = raw.pop("layout", None)
        for key, value in raw.items():
            if not hasattr(cfg, key):
                logger.warning("Ignoring unknown config key %r", key)
                continue
            if key == "background_color" and isinstance(value, str):
                value = hex_to_rgb(value)
            setattr(cfg, key, value)
        if isinstance(layout_raw, dict):
            for key, value in layout_raw.items():
                if hasattr(cfg.layout, key):
                    setattr(cfg.layout, key, value)
                else:
                    logger.warning("Ignoring unknown layout key %r", key)
        return cfg

    @classmethod
    def from_payload(cls, data: Dict[str, Any]) -> "KidsTriviaVideoBuilder":
        """One-stop factory: validated trivia dict -> configured builder."""
        return cls(TriviaQuestion.from_dict(data), cls.config_from_payload(data))

    # ------------------------------------------------------- primitive layers

    def _text_clip(self, text: str, font_size: int, color: RGB,
                   max_width: Optional[int] = None) -> TextClip:
        """Centered, word-wrapped TextClip (MoviePy v2 renders text via Pillow).

        ``margin=(12, 6)`` pads the text bitmap so the stroke outline and the
        descenders of glyphs ('g', 'y') are never clipped at the edges.
        """
        return TextClip(
            font=self._font_path,
            text=wrap_text(text, _pil_font(self._font_path, font_size),
                           max_width or self._px["text_max_width"]),
            font_size=font_size,
            color=tuple(int(c) for c in color),
            stroke_color=tuple(int(c) for c in self.config.stroke_color),
            stroke_width=self.config.stroke_width,
            interline=self.config.interline,
            text_align="center",
            margin=(12, 6),
            duration=self.config.duration,
        )

    def _centered_x(self, clip_w: int) -> int:
        """Horizontal centering: put the clip's LEFT edge so it sits mid-frame."""
        return (self.config.width - int(clip_w)) // 2

    def _card_layer(self, box: Box, rgba: Tuple[int, int, int, int],
                    layer_index: int, glow: bool = False,
                    duration: Optional[float] = None,
                    start: float = 0.0) -> ImageClip:
        """Rounded rectangle 'card' behind some text.

        ``box`` is (x, y, w, h) in pixels -- i.e. exactly what
        ``clip.with_position()`` expects for the TOP-LEFT corner.
        ``rgba`` colours the card; ``glow`` adds a blurred halo (reveal effect).
        """
        dur = self.config.duration if duration is None else duration
        x, y, w, h = (int(v) for v in box)
        base = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
        if glow:                                        # soft outer halo first
            halo = rounded_rect_image((w, h), self.config.card_radius,
                                      fill=min(255, int(rgba[3]) + 40),
                                      blur=max(6, h // 7))
            tinted = Image.new("RGBA", halo.size, rgba[:3] + (0,))
            tinted.putalpha(halo)
            base.alpha_composite(tinted)
        body = rounded_rect_image((w, h), self.config.card_radius, fill=rgba[3])
        solid = Image.new("RGBA", body.size, rgba[:3] + (0,))
        solid.putalpha(body)
        base.alpha_composite(solid)
        # RGBA numpy array -> ImageClip builds the RGB image AND its alpha mask
        clip = ImageClip(np.asarray(base, dtype=np.uint8), duration=dur)
        return (clip.with_start(start)
                    .with_position((x, y))
                    .with_layer_index(layer_index))

    def _solid_layer(self, box: Box, color: RGB, layer_index: int,
                     start: float = 0.0,
                     duration: Optional[float] = None,
                     opacity: float = 1.0) -> ColorClip:
        """Plain filled rectangle (the progress bar uses this as its 'asset')."""
        x, y, w, h = (int(v) for v in box)
        clip = ColorClip(size=(max(1, w), max(1, h)),
                         color=tuple(int(c) for c in color),
                         duration=self.config.duration if duration is None
                         else duration)
        clip = clip.with_start(start).with_position((x, y))
        if opacity != 1.0:
            clip = clip.with_opacity(opacity)
        return clip.with_layer_index(layer_index)

    def _badge_layer(self, box: Box, layer_index: int, start: float,
                     duration: float) -> ImageClip:
        """Green circle + white tick placed at (x, y, w, h) -- the answer stamp."""
        x, y, w, h = (int(v) for v in box)
        d = max(24, min(w, h))
        img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        pad = max(2, d // 14)
        draw.ellipse([pad, pad, d - pad, d - pad], fill=NEON_GREEN + (255,))
        lw = max(4, d // 9)
        pts = [(d * 0.28, d * 0.53), (d * 0.44, d * 0.69), (d * 0.73, d * 0.33)]
        for seg in range(len(pts) - 1):                 # two-stroke check mark
            draw.line([pts[seg], pts[seg + 1]], fill=(255, 255, 255, 255),
                      width=lw, joint="curve")
        clip = ImageClip(np.asarray(img, dtype=np.uint8), duration=duration)
        return (clip.with_start(start)
                    .with_position((x + (w - d) // 2, y + (h - d) // 2))
                    .with_layer_index(layer_index))

    # ------------------------------------------------------------ the phases

    def build_background(self) -> ColorClip:
        """Layer 0: solid pastel canvas filling the whole 1080x1920 frame."""
        bg = ColorClip(size=(self.config.width, self.config.height),
                       color=tuple(int(c) for c in self.config.background_color),
                       duration=self.config.duration)
        return bg.with_position((0, 0)).with_layer_index(0)

    def build_brand_and_footer(self) -> List[VideoClip]:
        """Tiny brand pill on top, CTA pinned near the bottom edge."""
        cfg, px = self.config, self._px
        clips: List[VideoClip] = []

        pill = self._text_clip(self.BRAND_LABEL, cfg.brand_font_size,
                               cfg.text_color)
        pw, ph = pill.size
        pad_x, pad_y = 26, 10
        clips.append(self._card_layer(
            (self._centered_x(pw) - pad_x, px["brand_top"] - pad_y,
             pw + 2 * pad_x, ph + 2 * pad_y),
            (0, 0, 0, 70), layer_index=1))
        clips.append(pill.with_position((self._centered_x(pw), px["brand_top"]))
                          .with_layer_index(1))

        cta = self._text_clip(self.CTA_LABEL, int(cfg.brand_font_size * 0.9),
                              cfg.text_color)
        # anchored by its BOTTOM edge:  y = y_bottom - clip_height
        clips.append(cta.with_position((self._centered_x(cta.size[0]),
                                        px["cta_bottom"] - cta.size[1]))
                          .with_layer_index(7))
        return clips

    def build_question_block(self) -> List[VideoClip]:
        """Question card near the TOP; visible for the entire video (0 -> 10 s)."""
        cfg, px = self.config, self._px
        txt = self._text_clip(self.question.question, cfg.question_font_size,
                              cfg.text_color)
        tw, th = txt.size
        pad_x, pad_y = 30, 26
        card_h = min(int(th + 2 * pad_y), int(cfg.height * 0.30))
        card_w = min(int(tw + 2 * pad_x), int(px["text_max_width"]))
        x = self._centered_x(card_w)
        # TOP-edge anchoring: the card starts at `question_top`, text centred in it
        y_text = px["question_top"] + max(0, (card_h - int(th)) // 2)
        return [
            self._card_layer((x, px["question_top"], card_w, card_h),
                             (0, 0, 0, cfg.card_body_alpha), layer_index=2),
            txt.with_position((x + (card_w - int(tw)) // 2, y_text))
               .with_layer_index(2),
        ]

    def build_option_blocks(self) -> List[VideoClip]:
        """Option cards stacked vertically around ``options_center_pct``.

        The stack is measured first, then shifted so its MIDDLE lands on the
        anchor line -- add/remove options and everything stays balanced.
        Returns every layer needed by BOTH phases: white text (0-10 s), neon
        green duplicate + glow + check badge (5-10 s), dimmed wrong answers.
        """
        cfg, px = self.config, self._px
        texts = [self._text_clip(opt, cfg.option_font_size, cfg.text_color)
                 for opt in self.question.options]
        widest = max(int(t.size[0]) for t in texts)
        tallest = max(int(t.size[1]) for t in texts)
        pad_x, pad_y = 34, 20
        badge_d = max(28, int(cfg.height * 0.045))   # check-mark stamp diameter
        card_w = min(widest + 2 * (pad_x + badge_d), int(cfg.width * 0.92))
        card_h = tallest + 2 * pad_y
        gap = px["option_gap"]
        total = len(texts) * card_h + (len(texts) - 1) * gap
        stack_top = px["options_center"] - total // 2      # centre anchoring
        x_card = self._centered_x(card_w)

        clips: List[VideoClip] = []
        for i, txt in enumerate(texts):
            y_card = stack_top + i * (card_h + gap)
            # text centred inside its own card (both axes)
            pos = (x_card + (card_w - int(txt.size[0])) // 2,
                   y_card + (card_h - int(txt.size[1])) // 2)
            placed = txt.with_position(pos).with_layer_index(3)
            if i == self.question.correct_index:
                # the winning card brightens up at the reveal instead of dimming
                clips.append(self._brighten_after_reveal(
                    (x_card, y_card, card_w, card_h)))
                clips += self._reveal_correct_answer(placed, pos, x_card,
                                                     y_card, card_w, card_h)
            else:
                clips.append(self._card_layer(
                    (x_card, y_card, card_w, card_h),
                    (0, 0, 0, cfg.card_body_alpha), layer_index=3))
                clips.append(self._dim_after_reveal(placed))
        return clips

    def _brighten_after_reveal(self, card_box: Box) -> VideoClip:
        """Winner's card: solid white plate fading in under the green text."""
        cfg = self.config
        plate = self._card_layer(card_box, (255, 255, 255, 235), layer_index=6,
                                 duration=cfg.reveal_duration,
                                 start=cfg.reveal_time)
        return attach_animated_mask(
            plate, lambda t, m: m * min(1.0, t / max(0.05, cfg.reveal_fade_in)))

    def _dim_after_reveal(self, clip: VideoClip) -> VideoClip:
        """Wrong answer: keeps its colour, fades to ``dimmed_opacity`` at 5 s."""
        cfg = self.config

        def dim_fn(t: float, m: np.ndarray) -> np.ndarray:
            ramp = min(1.0, max(0.0, t - cfg.reveal_time)
                       / max(0.05, cfg.reveal_fade_in))
            return m * (1.0 - (1.0 - cfg.dimmed_opacity) * ramp)

        return attach_animated_mask(clip, dim_fn)

    def _reveal_correct_answer(self, clip: VideoClip, pos: Tuple[int, int],
                               x_card: int, y_card: int, card_w: int,
                               card_h: int) -> List[VideoClip]:
        """Correct answer: white text until 5 s, neon green (+glow+tick) after.

        Two independent layers share the SAME (x, y) so the swap at the cut is
        invisible; the green one simply has ``duration = reveal_duration`` and
        ``start = reveal_time`` -- that is what makes the bar/digits 'disappear'
        too: their lifespan ends exactly at the 5-second mark.
        """
        cfg = self.config
        out: List[VideoClip] = [clip]     # white answer visible 0 -> 5 s

        green = self._text_clip(clip.text, cfg.option_font_size, NEON_GREEN)
        green = (green.with_duration(cfg.reveal_duration)
                      .with_start(cfg.reveal_time)
                      .with_position(pos)
                      .with_layer_index(6))
        out.append(green)

        if cfg.show_check_badge:          # tick stamped on the LEFT of the card
            out.append(self._badge_layer(
                (x_card + 14, y_card, card_h, card_h), layer_index=6,
                start=cfg.reveal_time, duration=cfg.reveal_duration))

        glow = self._card_layer(
            (x_card, y_card, card_w, card_h),
            NEON_GREEN + (120,), layer_index=6, glow=True,
            duration=cfg.reveal_duration, start=cfg.reveal_time)
        out.append(attach_animated_mask(
            glow, lambda t, m: m * min(1.0, t / max(0.05, cfg.reveal_fade_in))))
        return out

    def build_countdown(self) -> List[VideoClip]:
        """Progress bar (shrinks 100% -> 0%) + digits, live only for 0 -> 5 s."""
        cfg, px = self.config, self._px
        bar_w, bar_h = px["bar_full_width"], px["bar_height"]
        x0 = self._centered_x(bar_w)
        # the 'asset' is a plain rectangle: a dark track + a neon fill on top of it
        track = self._solid_layer((x0, px["bar_top"], bar_w, bar_h),
                                  (0, 0, 0), layer_index=4, opacity=0.25)
        fill = self._solid_layer((x0, px["bar_top"], bar_w, bar_h),
                                 NEON_GREEN, layer_index=4,
                                 duration=cfg.reveal_time)   # gone at t = 5 s

        def shrink_fn(t: float, m: np.ndarray) -> np.ndarray:
            """Left-anchored wipe: keep the leftmost ``progress`` columns opaque.

            Because the bar is horizontally centered, shrinking from the RIGHT
            edge reads as 'time running out' while the track stays put.
            """
            progress = 1.0 - min(1.0, max(0.0, t / max(0.001, cfg.reveal_time)))
            out = np.zeros_like(m)
            keep = int(round(m.shape[1] * progress))
            if keep > 0:
                out[:, :keep] = m[:, :keep]
            return out

        fill = attach_animated_mask(fill, shrink_fn)

        ticks = int(max(1, math.floor(cfg.reveal_time)))
        digits: List[VideoClip] = []
        for step in range(ticks):                 # digit 5 at t=0 ... digit 1 last
            label = str(ticks - step)
            num = self._text_clip(label, cfg.countdown_font_size, cfg.text_color)
            num = num.with_duration(cfg.duration).with_start(step * 1.0)
            num = num.with_position((self._centered_x(num.size[0]),
                                     px["counter_center"] - num.size[1] // 2))
            num = num.with_layer_index(5)

            def pulse(t: float, m: np.ndarray, s: float = step * 1.0) -> np.ndarray:
                local = t - s                      # seconds since this digit appeared
                if local < 0:
                    return m
                appear = min(1.0, local / 0.18)
                vanish = 1.0 if local < 0.75 else max(0.0, 1.0 - (local - 0.75) / 0.25)
                bounce = 1.0 + 0.10 * math.sin(min(local, 0.30) * 2 * math.pi)
                return np.clip(m * appear * vanish * bounce, 0.0, 1.0)

            digits.append(attach_animated_mask(num, pulse))
        return [track, fill] + digits

    def build_reveal_overlays(self) -> List[VideoClip]:
        """Extras that only exist during 5 -> 10 s (e.g. the 'TIME'S UP!' banner)."""
        cfg, px = self.config, self._px
        if not cfg.show_times_up_banner or cfg.reveal_duration <= 0:
            return []
        size = int(cfg.countdown_font_size * 0.72)
        banner = self._text_clip("TIME'S UP!", size, NEON_GREEN)
        bw, bh = int(banner.size[0]), int(banner.size[1])
        pad_x, pad_y = 34, 16
        card = self._card_layer(
            (self._centered_x(bw) - pad_x,
             px["timesup_center"] - bh // 2 - pad_y,
             bw + 2 * pad_x, bh + 2 * pad_y),
            (0, 0, 0, 165), layer_index=7,
            duration=cfg.reveal_duration, start=cfg.reveal_time)
        txt = banner.with_duration(cfg.reveal_duration).with_start(cfg.reveal_time)
        txt = txt.with_position((self._centered_x(bw),
                                 px["timesup_center"] - bh // 2)).with_layer_index(7)

        def pop_fn(t: float, m: np.ndarray) -> np.ndarray:
            scale = 0.85 + 0.15 * min(1.0, t / 0.22)   # quick grow-in pop
            return np.clip(m * scale, 0.0, 1.0)

        return [attach_animated_mask(card, pop_fn), attach_animated_mask(txt, pop_fn)]

    # ------------------------------------------------------------- assembly

    def build_clip(self) -> CompositeVideoClip:
        """Compose every layer into the final 9:16 clip (nothing written yet)."""
        cfg = self.config
        layers: List[VideoClip] = [self.build_background()]
        layers += self.build_brand_and_footer()
        layers += self.build_question_block()
        layers += self.build_option_blocks()
        layers += self.build_countdown()
        layers += self.build_reveal_overlays()

        comp = CompositeVideoClip(layers, size=(cfg.width, cfg.height))
        comp = comp.with_duration(cfg.duration)
        audio = build_audio_track(cfg)
        if audio is not None:
            comp = comp.with_audio(audio)
        logger.info("Composited %d layers -> %.1fs @ %dfps",
                    len(layers), cfg.duration, cfg.fps)
        return comp

    def render(self, output_path: str = "kids_trivia_short.mp4") -> RenderResult:
        """Export a compressed, web-optimized H.264 .mp4 (30 fps, yuv420p)."""
        cfg = self.config
        directory = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(directory, exist_ok=True)
        clip = self.build_clip()
        clip.write_videofile(
            output_path,
            fps=cfg.fps,
            codec=cfg.codec,                    # libx264 -> H.264
            preset=cfg.preset,
            audio_codec="aac" if cfg.audio_enabled else None,
            ffmpeg_params=[
                "-crf", str(cfg.crf),           # constant rate factor (quality)
                "-pix_fmt", cfg.pix_fmt,        # browser/mobile compatibility
                "-profile:v", cfg.profile,
                "-maxrate", cfg.maxrate,
                "-bufsize", cfg.bufsize,
                "-movflags", "+faststart",      # progressive web playback
                "-threads", str(cfg.threads),
            ],
            logger=None,
        )
        clip.close()
        result = RenderResult(
            output_path=os.path.abspath(output_path),
            size_bytes=os.path.getsize(output_path),
            duration=cfg.duration, width=cfg.width, height=cfg.height, fps=cfg.fps,
        )
        logger.info("Rendered %s (%.2f MB)", result.output_path,
                    result.size_bytes / 1_048_576)
        return result


# --------------------------------------------------------------------------- #
# Manual smoke test:  python kids_trivia_builder.py
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    DUMMY_QUESTION: Dict[str, Any] = {
        "question": "Which animal is the tallest in the world?",
        "options": ["A) Elephant", "B) Giraffe", "C) Blue Whale"],
        "correct_answer": "B) Giraffe",
    }

    OUTPUT_FILE = "output/kids_trivia_short.mp4"

    print("=" * 64)
    print(" Kids Trivia Video Builder - rendering a 9:16 sample short")
    print("=" * 64)

    builder = KidsTriviaVideoBuilder.from_payload(DUMMY_QUESTION)
    info = builder.render(OUTPUT_FILE)

    print(f"\nDone -> {info.output_path}")
    print(f"     {info.width}x{info.height} | {info.duration:.1f}s | "
          f"{info.fps} fps | H.264 | {info.size_bytes / 1_048_576:.2f} MB")

