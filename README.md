# Kids Trivia Video Builder — Vercel Deployment

Backend engine (`kids_trivia_builder.py`) + FastAPI serverless wrapper (`api/index.py`)
deployed to **Vercel** with the `@vercel/python` runtime.

## Project layout

```
.
├── api/
│   └── index.py         # Vercel serverless function (FastAPI app, maxDuration = 300)
├── public/
│   └── index.html       # Static demo UI served at /
├── kids_trivia_builder.py  # Core MoviePy rendering engine (imported by the API)
├── requirements.txt     # Installed automatically by @vercel/python
├── vercel.json          # Build/route/function config
└── .gitignore
```

## Deploy

### Option A — Dashboard (recommended)
1. Push this repo to GitHub/GitLab/Bitbucket.
2. On [vercel.com/new](https://vercel.com/new) import the repo.
3. Vercel auto-detects Python via `requirements.txt` + `api/`. No build command
   or output dir needed. Click **Deploy**.

### Option B — CLI
```bash
npm i -g vercel
vercel            # first run: link/create project
vercel --prod     # production deploy
```

## Verify the deployment

```bash
curl https://YOUR-PROJECT.vercel.app/health
# -> {"status":"ok"}

curl -X POST https://YOUR-PROJECT.vercel.app/api/generate \
     -H 'Content-Type: application/json' \
     -d '{"question":"Which animal is the tallest in the world?",
          "options":["A) Elephant","B) Giraffe","C) Blue Whale"],
          "correct_answer":"B) Giraffe","profile":"draft"}' \
     -o short.mp4 && ffprobe -v error -show_entries stream=codec_name,width,height -of csv short.mp4
# -> h264,540,960  (use "profile":"full" for 1080x1920)
```

Open `https://YOUR-PROJECT.vercel.app/` for the demo form, or `/docs` for Swagger UI.

## Endpoints

| Method | Path                  | Description                                    |
|--------|-----------------------|------------------------------------------------|
| GET    | `/`                   | Static demo UI (public/index.html)             |
| GET    | `/health`             | Liveness probe                                 |
| POST   | `/api/generate`       | Render MP4 from trivia JSON, returns video bytes |
| GET    | `/api/generate/sample?profile=draft` | Browser-friendly smoke test |
| GET    | `/docs`               | FastAPI Swagger docs                           |

Request body (extra `"config"` object can override any `VideoConfig`/`LayoutConfig`
field, e.g. `background_color`, `fps`, `layout.options_center_pct`):

```json
{
  "question": "Which animal is the tallest in the world?",
  "options": ["A) Elephant", "B) Giraffe", "C) Blue Whale"],
  "correct_answer": "B) Giraffe",
  "profile": "draft",
  "config": { "background_color": "#A8D8EA" }
}
```

## Platform limits & how they are handled

* **Read-only filesystem** → renders go to `/tmp` (`TRIVIA_OUTPUT_DIR=/tmp` set in `vercel.json`).
* **Function timeout** → `maxDuration: 300` s and `memory: 1024` MB declared in
  `vercel.json` (Pro plan; Hobby caps at 10 s). For Hobby accounts use
  `"profile": "draft"` (540×960, `veryfast` preset) which typically renders in
  a few seconds.
* **ffmpeg** → provided by the `imageio-ffmpeg` wheel (static binary), so no
  system install is required on the serverless image.
* **Fonts** → DejaVuSans-Bold ships on Vercel's Python runtime image; if you
  ever need a custom font, commit a `.ttf` and pass
  `"config": {"font_path": "assets/MyFont.ttf"}` (include it in the bundle).

## Local development

```bash
pip install -r requirements.txt
python kids_trivia_builder.py                 # render sample to output/
uvicorn api.index:app --reload --port 8000    # full API locally
vercel dev                                    # emulate the Vercel runtime
```

## Scaling note (production advice)

Serverless functions are not ideal for long CPU-bound encodes. For a real SaaS,
keep `api/index.py` as the thin HTTP layer but move heavy renders to a queue +
worker (e.g. Vercel Cron/queue → AWS Lambda/ECS or a RunPod worker) and upload
results to Vercel Blob/S3/R2, returning a signed URL instead of bytes. The
engine's `KidsTriviaVideoBuilder.render()` API is unchanged for that migration.
