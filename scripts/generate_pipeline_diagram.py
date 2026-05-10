"""Generate the AI Video Pipeline infographic poster via GPT-Image-2.

Single-call helper. Output goes to assets/pipeline_diagram.png. Used as the
hero image in the README. Re-run only when the design needs to change.

Usage:
  REPLICATE_API_TOKEN=... python3 scripts/generate_pipeline_diagram.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import replicate_helpers as base


ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "assets" / "pipeline_diagram.png"
LOG_PATH = ROOT / "assets" / "generate_pipeline_diagram.log"
IMAGE_MODEL = "openai/gpt-image-2"


PROMPT = """A clean composition of SEVEN MACOS TERMINAL WINDOWS arranged in a
grid on a deep matte black desktop wallpaper. Each window is a faithful macOS
Terminal.app rendering: rounded corners, subtle drop shadow, dark charcoal
title bar at top, THREE TRAFFIC-LIGHT DOTS in the title bar (red, yellow,
green from left to right) in their correct macOS positions, with the window's
title centered in the title bar in pale white text. The body of each window
is pure black with monospaced terminal text in glowing cyan exactly hex
#24bce3, clean and legible (think Berkeley Mono / IBM Plex Mono / JetBrains
Mono), crisp not pixelated, with a faint CRT phosphor glow on the cyan text.

LAYOUT (portrait 2:3 canvas):

ROW 1 (one wide window spanning full width):
  Window title bar: "ai-video-pipeline -- README.md"
  Window content (large heading text):
    HOW TO BUILD AN AI-GENERATED SHORT FILM
    A reproducible workflow. Storyboard -> animation -> final cut.
    Every prompt in source. Every artifact in a manifest.

ROW 2 (two side-by-side windows):
  LEFT WINDOW title bar: "the-stack.txt"
  Body:
    > 1. STORYBOARD    GPT-Image-2
    > 2. ANIMATION     Kling 3 Omni
    > 3. FINAL CUT     ffmpeg

  RIGHT WINDOW title bar: "the-numbers.txt"
  Body:
    > 28 scene panels
    > 13 video clips
    > 42 API calls
    > ~$22 total cost
    > ~40 min wall-clock
    > 72 sec final film

ROW 3 (two side-by-side windows):
  LEFT WINDOW title bar: "clip-patterns.txt"
  Body:
    > SINGLE-SHOT      one panel -> one clip
    > MULTI-PROMPT     up to 6 sub-shots in one call
    > KEYFRAME-BRIDGE  start_image + end_image

  RIGHT WINDOW title bar: "prompt-patterns.txt"
  Body:
    > character bible as input_images
    > strict negative-style headers
    > story-physics rule + override
    > multi_prompt < 512 chars
    > camera direction stated hard

ROW 4 (two side-by-side windows):
  LEFT WINDOW title bar: "workflow.sh"
  Body (terminal command flow with arrows):
    $ story idea
      |
      v
    [panel prompts] -> [character bible]
      |
      v
    [N scene panels] -> [video clips]
      |
      v
    [stitch + mix] -> [final.mp4]

  RIGHT WINDOW title bar: "what-went-wrong.log"
  Body:
    > i2v shadow physics ignored text
    > -f concat dropped video
    > amix halved audio
    > multi_prompt 511 char cap
    > --force regenerated bible

ROW 5 (one wide window spanning full width):
  Window title bar: "bring-your-own.txt"
  Body:
    > logo PNG  ->  assets/logos/Symbol_Cyan.png
    > brand color hex
    > 28 panel prompts (your story)
    > brand name + tagline
    github.com/0xadvait/ai-video-pipeline    MIT

VISUAL RULES:
- Every window has the same chrome aesthetic (rounded corners, thin border,
  charcoal title bar, traffic-light dots, faint window shadow against black)
- ALL body text is cyan #24bce3 on pure black, monospaced, crisp
- Title bar text is pale grey-white
- Spacing between windows is clean and balanced — like an organized desktop,
  not chaotic
- No other UI elements, no other colors except: traffic-light dots, cyan body
  text, pale title bar text, deep black backgrounds
- Portrait 2:3 aspect ratio overall
- This should look exactly like a screenshot of someone's workspace with
  seven Terminal windows tiled on a dark monitor"""


def main() -> int:
    base.setup_logging("pipeline_diagram", LOG_PATH)
    import logging
    logger = logging.getLogger("pipeline_diagram")
    client = base.load_replicate_client()
    fields = base.schema_fields(client, IMAGE_MODEL, logger)

    payload = {
        "prompt": PROMPT,
        "aspect_ratio": "2:3",
        "quality": "high",
        "output_format": "png",
        "num_images": 1,
        "num_outputs": 1,
        "number_of_images": 1,
    }
    payload = base.keep_supported_fields(payload, fields)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = base.run_replicate(client, IMAGE_MODEL, payload, OUT_PATH, logger,
                                 dry_run=False, repo_root=ROOT, rate_limit=False)
    if result["status"] != "ok":
        logger.error("FAIL: %s", result.get("error", "unknown"))
        return 1
    logger.info("OK %.1fMB -> %s", result["size_bytes"] / 1_000_000, OUT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
