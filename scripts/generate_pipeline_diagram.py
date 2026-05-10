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


PROMPT = """An information-dense technical infographic poster in a dark terminal
aesthetic. Portrait orientation. Deep matte black background. ALL text rendered
in glowing cyan exactly hex #24bce3 on black, in a clean monospaced terminal
font (IBM Plex Mono / JetBrains Mono / Berkeley Mono register), crisp and
legible, slight CRT phosphor glow. Light cyan grid lines faintly visible behind
content. Bordered sections with thin cyan rectangle outlines. Bullet items
marked with ">" character. Slight scanline texture overlay for terminal vibe.

TOP-LEFT CORNER: three small colored dots (red, yellow, green) like a macOS
window bar.

TOP-CENTER TITLE in large all-caps stencil-monospaced cyan text:
"HOW TO BUILD AN AI-GENERATED SHORT FILM"

DIRECTLY UNDER TITLE in smaller cyan terminal text, two short lines:
"A reproducible workflow.  Storyboard -> animation -> final cut."
"Every prompt in source. Every artifact in a manifest."

THE POSTER BODY IS DIVIDED INTO BORDERED SECTIONS:

SECTION 1 (top-left box) header "THE STACK":
> 1. STORYBOARD   GPT-Image-2  (stills)
> 2. ANIMATION    Kling 3 Omni (motion)
> 3. FINAL CUT    ffmpeg       (concat + mix)

SECTION 2 (top-right box) header "THE NUMBERS":
> 28 scene panels
> 13 video clips
> 42 API calls
> ~$22 total cost
> ~40 min wall-clock
> 72 sec final film

SECTION 3 (mid-left box) header "THREE CLIP PATTERNS":
> SINGLE-SHOT     one panel  ->  one motion clip
> MULTI-PROMPT    up to 6 sub-shots in one call
> KEYFRAME-BRIDGE start_image + end_image

SECTION 4 (mid-right box) header "THE PROMPT PATTERNS":
> character bible as input_images
> strict negative-style headers
> story-physics rule + override syntax
> multi_prompt < 512 chars per shot
> camera direction stated hard

SECTION 5 (lower-left box) header "WORKFLOW":
A simple flowchart with labeled arrows showing the data flow:
[story idea] -> [panel prompts] -> [character bible]
              -> [N scene panels]  -> [video clips]
              -> [stitch + mix]    -> [final.mp4]

SECTION 6 (lower-right box) header "WHAT WENT WRONG":
> i2v shadow physics ignored text
> -f concat dropped video
> amix halved audio
> multi_prompt 511 char cap
> --force regenerated bible

BOTTOM-CENTER  large box header "BRING YOUR OWN":
> logo PNG
> brand color hex
> 28 panel prompts (your story)
> brand name + tagline

BOTTOM-RIGHT CORNER small text:
"github.com/0xadvait/ai-video-pipeline   MIT"

Visual style: cinematic dark terminal poster, like a developer's
documentation render, very dense but every line of text legible. Monospaced
font throughout. Cyan-on-black ONLY (no other colors except the three
window dots in the corner). Print-poster proportions, portrait 2:3."""


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
