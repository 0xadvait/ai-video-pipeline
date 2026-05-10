"""Multi-shot Kling Omni call for the continuous walking arc 15→16→17.

Three shots × 2 sec = 6-second continuous beat where the agent never stops
walking and the world transforms around him:

  15 — walks along a thin cyan seam through a dim city
  16 — same walk, the seam now embedded with brightening brand-mark fragments
  17 — same walk, the pavement peels open and the city becomes translucent

Each shot prompt is intentionally tight (single rich sentence) following
the pattern in user-supplied examples (fire-tree-galaxy multi_prompt). No
micromanage of physics or audio per shot — those go in the top-level prompt.

Usage:
  REPLICATE_API_TOKEN=... python3 generate_multi_15_16_17.py
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
OUT_PATH = TRANS_DIR / "multi_15_16_17.mp4"
LOG_PATH = TRANS_DIR / "generate_multi_15_16_17.log"
KLING_OMNI = "kwaivgi/kling-v3-omni-video"


def setup_logging() -> logging.Logger:
    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multi_15_16_17")
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


# Tight, visual, one-sentence-each prompts. Continuous walk; the agent
# never stops, the world transforms around him.
SHOT_15 = (
    "Use <<<image_1>>>: the small white humanoid AI agent walks forward "
    "along a thin glowing cyan seam stretching through a dim wet "
    "desaturated city street, his bare feet leaving no shadow on the "
    "pavement, the seam pulsing softly cyan as he steps over each "
    "section, slow patient stride."
)

SHOT_16 = (
    "Use <<<image_2>>>: the same agent keeps walking the same seam, now "
    "embedded with fragmented [Your Brand] glyphs (4-fold rotational "
    "marks with rounded petal-lobes around a center dot) that glow more "
    "brilliantly as his bare feet pass over them, the dystopian city "
    "architecture around him subtly stretching upward into translucent "
    "geometry."
)

SHOT_17 = (
    "Use <<<image_3>>>: the agent continues walking forward as the "
    "pavement peels open along the seam revealing glowing cyan circuit-"
    "board layers beneath, the buildings stretching up and turning "
    "translucent in cool cyan around him, the agent dwarfed at the "
    "heart of a vast unfolding world."
)


def main() -> int:
    logger = setup_logging()
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise SystemExit("REPLICATE_API_TOKEN required.")
    import replicate

    client = replicate.Client(api_token=token)
    fields = base.schema_fields(client, KLING_OMNI, logger)

    p15 = next(p for p in cb.PANELS if p.idx == 15)
    p16 = next(p for p in cb.PANELS if p.idx == 16)
    p17 = next(p for p in cb.PANELS if p.idx == 17)
    for p in (p15, p16, p17):
        if not p.path.exists():
            raise SystemExit(f"Missing panel: {p.path.relative_to(ROOT)}")

    top_prompt = (
        "A continuous slow-walking journey of the small humanoid AI agent "
        "through a transforming dystopian city, with three visual states "
        "as the world around him changes — but his walking pace and "
        "direction never change. Reference images: shot 1 = <<<image_1>>>, "
        "shot 2 = <<<image_2>>>, shot 3 = <<<image_3>>>. Cinematic semi-"
        "photoreal CG, 1080p Pro, brand cyan #24bce3 the only saturated "
        "color, otherwise desaturated city greys. The protagonist agent "
        "casts NO shadow on any surface in any of these shots — the "
        "pavement beneath him is luminous and clean. Audio: a continuous "
        "rich harmonic ambient drone building from cyan resonance, soft "
        "rhythmic footsteps of the agent's bare feet, deep distant rumble "
        "swelling toward the world unfolding at the end."
    )

    multi_prompt = [
        {"prompt": SHOT_15, "duration": 2},
        {"prompt": SHOT_16, "duration": 2},
        {"prompt": SHOT_17, "duration": 2},
    ]
    for i, shot in enumerate(multi_prompt):
        if len(shot["prompt"]) >= 512:
            raise SystemExit(f"shot {i} prompt is {len(shot['prompt'])} chars (max 511)")

    payload: dict = {
        "prompt": top_prompt,
        "start_image": p15.path,
        "reference_images": [p15.path, p16.path, p17.path],
        "multi_prompt": json.dumps(multi_prompt),
        "mode": "pro",
        "generate_audio": True,
        "duration": 6,
    }
    payload = base.keep_supported_fields(payload, fields)

    logger.info("RUN multi-shot 15/16/17 -> %s", OUT_PATH.relative_to(ROOT))
    result = base.run_replicate(client, KLING_OMNI, payload, OUT_PATH, logger, dry_run=False, ROOT)
    if result["status"] != "ok":
        logger.error("FAIL: %s", result.get("error", "unknown"))
        return 1
    logger.info("OK %.1fMB", result["size_bytes"] / 1_000_000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
