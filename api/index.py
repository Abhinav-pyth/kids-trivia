"""
Vercel Python serverless entry point -- "Kids Trivia Video Builder" API
=======================================================================

Vercel maps every file inside ``/api`` to a serverless function.  This module
exposes a small FastAPI app (served through Vercel's Python runtime) with:

    GET  /            -> interactive docs are at /docs; root returns service info
    GET  /health      -> liveness probe
    POST /api/generate-> render one 9:16 trivia video, return it as MP4 bytes

The heavy lifting lives in ``kids_trivia_builder.KidsTriviaVideoBuilder``;
this file only adapts it to a request/response cycle.

IMPORTANT -- Vercel platform limits you must know about for video rendering:
  * Serverless filesystem is READ-ONLY except ``/tmp``  -> we always write to
    a temp dir (see ``_render_to_temp``).
  * Hobby plan functions time out after 10 s (Pro: 300 s, configurable via
    ``maxDuration`` in this file / vercel.json).  A 1080x1920 x 10 s render is
    usually 20-60 s, so either run on Pro with ``maxDuration = 300`` or use the
    built-in draft profile (``"profile": "draft"`` in the request body) which
    renders at 540x960 / veryfast and comfortably fits short timeouts.
  * Response bodies are capped (1024 MB, but keep them small); we stream the
    finished MP4 back directly and also support ``?mode=url`` style clients by
    returning proper Content-Disposition headers.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import uuid
from typing import Any, Dict

# Make the repo root importable both locally (uvicorn) and on Vercel, where
# the function bundle may place this file at the top of the module search path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field, ValidationError  # noqa: E402

try:  # normal case: kids_trivia_builder.py sits next to /api in the repo root
    from kids_trivia_builder import KidsTriviaVideoBuilder, VideoConfig
except ModuleNotFoundError:  # pragma: no cover - Vercel bundled-layout fallback
    from kids_trivia_builder import KidsTriviaVideoBuilder, VideoConfig  # type: ignore

logger = logging.getLogger("kids_trivia.api")

# Vercel sets maxDuration for Python serverless functions when declared here.
# 300 s is the Pro-plan ceiling; hobby accounts should lower it or use drafts.
max_duration = 300  # seconds

app = FastAPI(
    title="Kids Trivia Video Builder",
    description="Generates 9:16 (1080x1920) YouTube-Shorts trivia videos.",
    version="1.0.0",
)


# --------------------------------------------------------------------------- #
# Request schema
# --------------------------------------------------------------------------- #

class TriviaRequest(BaseModel):
    """Body of POST /api/generate -- mirrors the engine's input dict plus an
    optional ``config`` block (any VideoConfig/LayoutConfig field) and a
    ``profile`` shortcut for fast drafts."""

    question: str = Field(..., min_length=1, examples=["Which animal is the tallest in the world?"])
    options: list[str] = Field(..., min_length=2, examples=[["A) Elephant", "B) Giraffe", "C) Blue Whale"]])
    correct_answer: str = Field(..., min_length=1, examples=["B) Giraffe"])
    profile: str = Field("full", pattern="^(full|draft)$",
                         description="'draft' = 540x960 veryfast (fits tight timeouts)")
    config: Dict[str, Any] = Field(default_factory=dict,
                                   description="Optional VideoConfig overrides, e.g. {'background_color': '#A8D8EA', 'layout': {...}}")


def _payload_with_profile(req: TriviaRequest) -> Dict[str, Any]:
    """Convert the validated request into the raw dict the engine expects,
    applying the 'draft' performance profile when selected."""
    payload: Dict[str, Any] = {
        "question": req.question,
        "options": req.options,
        "correct_answer": req.correct_answer,
        "config": dict(req.config),
    }
    if req.profile == "draft":
        # Draft: quarter resolution + fastest preset => ~10x quicker encode.
        draft_defaults = {"width": 540, "height": 960, "preset": "veryfast", "crf": 26}
        for key, value in draft_defaults.items():
            payload["config"].setdefault(key, value)
    return payload


def _render_to_temp(payload: Dict[str, Any]) -> str:
    """Render the video into a throwaway temp file (Vercel FS is read-only
    outside /tmp) and return its path. Caller is responsible for cleanup."""
    builder = KidsTriviaVideoBuilder.from_payload(payload)
    out_dir = os.environ.get("TRIVIA_OUTPUT_DIR") or tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"trivia_{uuid.uuid4().hex}.mp4")
    builder.render(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({
        "service": "Kids Trivia Video Builder",
        "status": "ok",
        "endpoints": {
            "POST /api/generate": "Render a 9:16 trivia MP4 (returns video bytes)",
            "GET /api/generate/sample": "Render the built-in demo question",
            "GET /health": "Liveness probe",
            "GET /docs": "Interactive OpenAPI docs",
        },
    })


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate")
def generate(req: TriviaRequest):
    """Render one trivia video and stream it back as an MP4 download."""
    payload = _payload_with_profile(req)
    try:
        path = _render_to_temp(payload)
    except ValueError as exc:            # bad trivia data -> 422
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValidationError as exc:       # defensive; FastAPI usually catches first
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:             # ffmpeg/font/runtime failure -> 500
        logger.exception("render failed")
        raise HTTPException(status_code=500, detail=f"Render failed: {exc}") from exc

    return FileResponse(
        path,
        media_type="video/mp4",
        filename="kids_trivia_short.mp4",
        background=None,
    )


@app.get("/api/generate/sample")
def generate_sample(profile: str = "draft"):
    """Quick smoke test straight from the browser: GET /api/generate/sample"""
    sample = TriviaRequest(
        question="Which animal is the tallest in the world?",
        options=["A) Elephant", "B) Giraffe", "C) Blue Whale"],
        correct_answer="B) Giraffe",
        profile=profile,
    )
    return generate(sample)


# Local development helper: `python api/index.py` runs uvicorn on $PORT.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
