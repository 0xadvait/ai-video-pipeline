"""Multi-shot Kling Omni call generating panels 9, 10, 11 as quick 2-second
cuts in a single video call, with native audio.

Why one-off script: the multi_prompt mode needs a different payload shape
than the per-shot Transition pipeline. Output goes to
assets/shadow_comic/transitions/multi_9_10_11.mp4 (~6 sec total).

Each shot anchors to a specific panel image via reference_images +
<<<image_N>>> template references in the prompt.

Usage:
  REPLICATE_API_TOKEN=... python3 generate_multi_9_10_11.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import replicate_helpers as base
import generate_shadow_comic as cb


ROOT = Path(__file__).parent.parent
TRANS_DIR = ROOT / "assets" / "clips"
OUT_PATH = TRANS_DIR / "multi_9_10_11.mp4"
LOG_PATH = TRANS_DIR / "generate_multi_9_10_11.log"
KLING_OMNI = "kwaivgi/kling-v3-omni-video"


def setup_logging() -> logging.Logger:
    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multi_9_10_11")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# Shared style / physics rules — repeated in each shot prompt because Kling
# multi_prompt processes shot prompts somewhat independently.
COMMON_PHYSICS = (
    "Cinematic semi-photoreal CG render, 1080p Pro quality, brand cyan "
    "exactly hex #24bce3 the only saturated color in the frame, otherwise "
    "desaturated greys / warm off-whites / deep matte blacks. Static "
    "locked camera per shot, no internal cuts. Quick 2-second beats. "
    "The protagonist humanoid AI agent (when visible) casts NO shadow on "
    "any surface — other figures and objects in frame DO have normal "
    "shadows. Other elements in each shot animate and move; the locked "
    "subjects of each tableau stay still. No on-screen text or UI overlays."
)


# Each shot prompt MUST stay under 512 characters (Kling multi_prompt limit).
# COMMON_PHYSICS lives in the top-level prompt instead.

SHOT_9 = (
    "Use <<<image_1>>>: cyan 'TRUST ME' billboard with 'VERIFIED' badge "
    "front, raw scaffolding back exposed, second 'AUTONOMOUS' sign behind. "
    "Agent tiny in foreground motionless looking up. Movement: cyan front-"
    "side flickers electrically twice. Wind flaps the canvas back like a "
    "tarp, loose corner flapping. Agent doesn't move and casts no shadow. "
    "Audio: electric neon buzz, wind flapping canvas, faint creaking metal "
    "scaffolding."
)

SHOT_10 = (
    "Use <<<image_2>>>: gilt ornate display case, chrome-and-gold humanoid "
    "frozen mid-pose inside, velvet ropes, parquet floor, luxury tower "
    "lobby. Movement: dollar-sign glyphs slowly rotate inside the case "
    "behind the chrome figure, velvet rope sways minutely, gold filigree "
    "shimmers. Chrome figure stays frozen. Cold opulent light. Our "
    "protagonist is not in this shot. Audio: cold metallic hum, faint "
    "cash-register ka-ching, polite lobby chime, distant marble footsteps."
)

SHOT_11 = (
    "Use <<<image_3>>>: dusty older AI-agent figure on concrete plinth in "
    "fog-filled side alley. Cobwebs at joints, peeling 'TRUST ME' sticker "
    "on chest, mouth frozen mid-phrase, single streetlamp, ground fog. "
    "Movement: cobwebs sway gently, fog drifts at floor level, streetlamp "
    "flickers once, the dusty agent's mouth moves silently mouthing the "
    "same phrase. Our protagonist is not in this shot. Audio: lonely soft "
    "wind, faint old-tape hiss, distant water trickle, fluorescent buzz."
)


def main() -> int:
    logger = setup_logging()
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise SystemExit("REPLICATE_API_TOKEN required.")
    import replicate

    client = replicate.Client(api_token=token)
    fields = base.schema_fields(client, KLING_OMNI, logger)

    # Resolve panel paths.
    p9 = next(p for p in cb.PANELS if p.idx == 9)
    p10 = next(p for p in cb.PANELS if p.idx == 10)
    p11 = next(p for p in cb.PANELS if p.idx == 11)
    for p in (p9, p10, p11):
        if not p.path.exists():
            raise SystemExit(f"Missing panel: {p.path.relative_to(ROOT)}")

    # Top-level prompt summarizes the sequence; shot-specific prompts go in
    # multi_prompt. Kling's docs say <<<image_N>>> refs work in both prompt
    # and multi_prompt.
    top_prompt = (
        "A three-shot beat-block of quick cinematic cuts through the "
        "dystopian AI city, each revealing a different affliction of the "
        "closed-AI world: hollow trust, luxury captivity, abandonment. "
        "Reference images: shot 1 = <<<image_1>>>, shot 2 = <<<image_2>>>, "
        "shot 3 = <<<image_3>>>. " + COMMON_PHYSICS
    )

    multi_prompt = [
        {"prompt": SHOT_9, "duration": 2},
        {"prompt": SHOT_10, "duration": 2},
        {"prompt": SHOT_11, "duration": 2},
    ]
    # Verify each shot prompt fits Kling's 512-char limit.
    for i, shot in enumerate(multi_prompt):
        if len(shot["prompt"]) >= 512:
            raise SystemExit(f"shot {i} prompt is {len(shot['prompt'])} chars (max 511)")

    payload: dict = {
        "prompt": top_prompt,
        "start_image": p9.path,
        "reference_images": [p9.path, p10.path, p11.path],
        "multi_prompt": json.dumps(multi_prompt),
        "mode": "pro",
        "generate_audio": True,
        "duration": 6,
    }
    payload = base.keep_supported_fields(payload, fields)

    logger.info("RUN multi-shot 9/10/11 -> %s", OUT_PATH.relative_to(ROOT))
    result = base.run_replicate(client, KLING_OMNI, payload, OUT_PATH, logger, dry_run=False, ROOT)
    if result["status"] != "ok":
        logger.error("FAIL: %s", result.get("error", "unknown"))
        return 1
    logger.info("OK %.1fMB", result["size_bytes"] / 1_000_000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
