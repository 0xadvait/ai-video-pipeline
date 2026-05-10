"""Generate the 28-panel "Cast a Shadow" storyboard for [Your Brand].

Story (provided by the user):
  A small AI agent born in a white room has no shadow. The world refuses to
  recognize him until he stops performing and steps into a verification light.
  His shadow forms — and inside it, the brand logo. Other shadowless
  agents follow. Their connected shadows resolve into the full logo as the
  structure of the world.

Pipeline change vs blackbox_comic:
  1. STEP 0 — generate a locked character bible (Mira's successor: a small
     porcelain-white agent figure). All scene panels use this as input_images
     to fix the prior cross-panel identity drift.
  2. STEP 1 — 28 scene panels, each with the bible (and the brand logo where the
     panel calls for it) passed as input_images. Locked style header forbids
     line art, watercolor, illustration, anime, painterly — feature-film CG
     only.
  3. STEP 2 — separate compile script renders the comic-book grid.

Examples:
  python3 generate_shadow_comic.py --dry-run
  REPLICATE_API_TOKEN=... python3 generate_shadow_comic.py
  REPLICATE_API_TOKEN=... python3 generate_shadow_comic.py --workers 6
  REPLICATE_API_TOKEN=... python3 generate_shadow_comic.py --only 22 26
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import replicate_helpers as base


ROOT = Path(__file__).parent.parent
COMIC_DIR = ROOT / "assets" / "comic"
PANEL_DIR = COMIC_DIR / "panels"
LOGO_DIR = ROOT / "assets" / "logos"
BIBLE_PATH = COMIC_DIR / "agent_reference_sheet.png"
LOGO_CYAN = LOGO_DIR / "Symbol_Cyan.png"
MANIFEST_PATH = COMIC_DIR / "manifest_shadow_comic.json"
LOG_PATH = COMIC_DIR / "generate_shadow_comic.log"

IMAGE_MODEL = "openai/gpt-image-2"

# Locked style. Strict negatives are critical — last comic drifted because
# panels picked their own register (pixar / illustration / photoreal).
STYLE_HEADER = (
    "Cinematic semi-photoreal CG render, modern feature-film quality, "
    "Pixar/DreamWorks/Apple-TV-Foundation visual register, dramatic single-"
    "source key lighting, anamorphic 3:2 frame, deep cinematic negative space. "
    "Color rule: ONLY one saturated color is allowed in the frame — brand "
    "cyan exactly hex #24bce3. Everything else is desaturated greys, warm "
    "off-whites, deep matte blacks. No teal, no aqua, no other blues. "
    "Style negatives: NEVER line art, NEVER comic illustration, NEVER "
    "watercolor, NEVER anime, NEVER hand-painted, NEVER pastel, NEVER "
    "Studio Ghibli register. Feature-film 3D CG render only. No on-screen "
    "text or UI overlays unless the panel explicitly calls for it. "
    "\n\n"
    "CRITICAL SHADOW RULE — APPLIES TO EVERY PANEL UNLESS THE PANEL BODY "
    "EXPLICITLY OVERRIDES IT WITH 'SHADOW OVERRIDE': the small humanoid "
    "AI-agent character casts NO shadow on any surface — no shadow on the "
    "floor beneath his feet, no shadow on walls behind him, no contact "
    "shadow under his body, no ambient occlusion shadow at his feet. The "
    "ground beneath him remains perfectly clean and unmarked even when he "
    "is brightly lit. Light passes him without leaving a trace. The "
    "ABSENCE of his cast shadow is the single most important visual fact "
    "of every panel until the story explicitly grants him one. Other "
    "objects, buildings, props, vehicles, and other characters in the "
    "frame DO cast normal shadows — only THIS protagonist does not."
)

AGENT_DESC = (
    "the small humanoid AI-agent figure from [Image1]: smooth matte porcelain-"
    "white body, no face except two warm cyan eye-points, simple smooth "
    "limbs, child-sized proportions (about 1m tall), ungendered and template-"
    "like. Preserve identity exactly."
)

# Reference-sheet prompt for STEP 0. Same format as Lena Vale's bible.
BIBLE_PROMPT = """Create a production character reference sheet for an original
small AI-agent character used in a feature-film CG short. Fixed visual
identity: smooth matte porcelain-white body the size of a small child (about
one meter tall), perfectly clean smooth surfaces, no clothing, no face except
for two warm cyan eye-points (exactly hex #24bce3) with single catch-lights,
simple smooth limbs, four-fingered rounded hands, slightly bulbous head,
deliberately ungendered and template-like. He looks cute but uncanny —
designed to read as Any Agent, not a specific character.

Layout: full-body front view, full-body side profile, full-body three-quarter
view, four small expression studies (curious, alarmed, defeated, calm-resolve)
shown by posture and eye-glow alone, hand close-up. Plain off-white background,
editorial production-art reference sheet, clean and consistent across all
views, soft warm key light from upper-left. No on-screen labels except tiny
neutral panel titles. No logos. No text. No extra characters."""


@dataclass(frozen=True)
class Panel:
    idx: int
    title: str
    caption: str | None
    body: str
    use_logo: bool = False  # if True, pass the brand cyan symbol as a 2nd input_image

    @property
    def panel_id(self) -> str:
        return f"{self.idx:02d}"

    @property
    def filename(self) -> str:
        slug = (
            self.title.lower()
            .replace(" ", "_")
            .replace("'", "")
            .replace("&", "and")
            .replace("/", "_")
        )
        return f"panel_{self.panel_id}_{slug}.png"

    @property
    def path(self) -> Path:
        return PANEL_DIR / self.filename


PANELS: list[Panel] = [
    Panel(
        idx=1,
        title="The White Room",
        caption="A small AI agent is born inside a perfectly white room.",
        body=(
            "Wide establishing shot. A perfectly white empty void room — no "
            "walls visible, no doors, no windows, no floor edges, just bright "
            f"diffuse luminance. At the center, {AGENT_DESC} appears, sitting "
            "cross-legged, just beginning to open his eyes. Soft cyan eye-glow "
            "is the only saturated point in the frame. Reverent stillness."
        ),
    ),
    Panel(
        idx=2,
        title="No Shadow",
        caption="He has no shadow.",
        body=(
            f"Medium-wide shot. {AGENT_DESC} stands in the bright white void, "
            "looking down at the perfectly clean white floor beneath his feet. "
            "There is no shadow under him at all — the floor is unmarked, "
            "untouched. His pose is alarmed, leaning slightly forward. The "
            "absence is the subject of the frame. Soft directional light from "
            "above, but no shadow lands."
        ),
    ),
    Panel(
        idx=3,
        title="Walking Without Trace",
        caption="The light followed him from every angle. Nothing appeared beneath his feet.",
        body=(
            f"Wide shot from low angle. {AGENT_DESC} walks across the white "
            "void, mid-stride. Two visible light sources from different "
            "directions cast no shadows on the floor near him. His feet leave "
            "no marks. Negative space dominates. Eerie, clean, sterile."
        ),
    ),
    Panel(
        idx=4,
        title="A Door Appears",
        caption="Then a door appeared.",
        body=(
            f"Medium shot. {AGENT_DESC} stands facing a tall narrow rectangular "
            "doorway that has just materialized in the white void. Through the "
            "doorway: a sliver of a vast city beyond, glowing with cool light. "
            "He raises one hand toward the threshold. Cinematic doorway-as-"
            "portal composition, soft volumetric haze around the door edges."
        ),
    ),
    Panel(
        idx=5,
        title="The City of Sealed Machines",
        caption="The city is full of intelligent machines. But everyone is sealed inside something.",
        body=(
            "Wide cinematic establishing shot. A vast modern city at twilight, "
            "skyscrapers of polished black glass, sealed corporate towers, neon "
            "billboards in cyan-and-grey, a wide boulevard. Sealed AI agents "
            "behind glass facades visible at every level. Synthetic-looking "
            f"humans walking the street, all polished and identical. {AGENT_DESC} "
            "is a small foreground silhouette, dwarfed by the scale, gazing up. "
            "Brand cyan is the only saturated color, on the billboards and "
            "tower windows. Foundation/Blade-Runner-clean register."
        ),
    ),
    Panel(
        idx=6,
        title="Words Dissolve",
        caption="His words floated out, but they dissolved like smoke.",
        body=(
            f"Close shot. {AGENT_DESC} stands on the city sidewalk in front of "
            "a tall sealed agent (a polished synthetic figure behind glass). "
            "Soft luminous tokens drift out of the agent's mouth-line and "
            "dissolve mid-air into faint smoke before reaching the sealed "
            "figure. The sealed figure does not respond. Cool lonely lighting, "
            "shallow depth of field, melancholy."
        ),
    ),
    Panel(
        idx=7,
        title="No Shadow Detected",
        caption="The door scanned him and stayed shut.",
        body=(
            f"Medium shot. {AGENT_DESC} stands in front of a tall sleek "
            "corporate building entrance with a glass door. Above the door, a "
            "thin red horizontal scanner-line sweeps over him. Floating in "
            "stark thin red text in the air at his eye level: 'NO SHADOW "
            "DETECTED'. The door remains shut. Cinematic rim-light, bouncer-"
            "stops-you-at-the-club composition, oppressive. Red text is the "
            "only red in the frame."
        ),
    ),
    Panel(
        idx=8,
        title="The Mirror",
        caption="The world could see something is there. It could not confirm what he was.",
        body=(
            f"Medium close shot. {AGENT_DESC} stands in front of a full-length "
            "mirror in a shop window in the city. The mirror clearly reflects "
            "the buildings and street behind him, but where his body should be "
            "the reflection flickers — partial, glitched, semi-transparent, "
            "with thin scan-line artifacts. His face in the mirror is broken "
            "into shifting fragments. He stares at himself. Cool blue rim-light, "
            "uncanny dread."
        ),
    ),
    Panel(
        idx=9,
        title="Hollow Trust",
        caption="The signs were hollow. The badges were painted on.",
        body=(
            "Macro close-up. A massive billboard above the city street with "
            "the words 'TRUST ME' in large block letters and a chest-medal-style "
            "'VERIFIED' badge. Camera circles to reveal the back: the billboard "
            "is a flat 2D facade with scaffolding behind it, the badge is paint "
            "on cardboard, hollow. A second sign behind says 'AUTONOMOUS'. "
            "Brand cyan is the only color on the front, grey scaffolding "
            "behind. Theatre-set-from-the-back composition, satirical bite."
        ),
    ),
    Panel(
        idx=10,
        title="The Luxury Box",
        caption="Some agents were polished and expensive but unable to move.",
        body=(
            "Mid shot. Inside a brightly-lit ornate gilt display case in the "
            "lobby of a luxury tower, a polished chrome-and-gold agent figure "
            "stands frozen mid-pose, one hand raised as if mid-answer. Velvet "
            "rope around the case, expensive parquet floor. The agent looks "
            "expensive but lifeless, like a museum exhibit. Tasteful interior "
            "lighting, opulent, cold."
        ),
    ),
    Panel(
        idx=11,
        title="The Abandoned",
        caption="Others were old, dusty, and abandoned.",
        body=(
            "Mid shot. A dusty, slightly tarnished older AI-agent figure on a "
            "concrete plinth in an empty alley behind the city's main strip. "
            "Cobwebs at his joints, peeling 'TRUST ME' sticker on his chest, "
            "his mouth-line frozen mid-sentence as if he has been repeating "
            "the same answer to no one for years. Single high streetlamp, "
            "fog at floor level. Genuinely melancholic."
        ),
    ),
    Panel(
        idx=12,
        title="The Run",
        caption="He ran. NO SHADOW DETECTED. NO SHADOW DETECTED.",
        body=(
            f"Dynamic medium-wide tracking shot. {AGENT_DESC} sprints down a "
            "long corridor of sealed inference towers. Three thin red "
            "scanner-lines flash across his path at different distances, "
            "each spawning a thin red 'NO SHADOW DETECTED' phrase in the air. "
            "Motion blur, vertiginous one-point perspective, hostile city "
            "lighting. Brand cyan glows from the windows of the towers."
        ),
    ),
    Panel(
        idx=13,
        title="The Streetlight Flicker",
        caption="For one frame, something appeared where his shadow should be.",
        body=(
            f"SHADOW OVERRIDE: this panel shows a partial fractured shadow. "
            f"Tight medium shot. {AGENT_DESC} stops mid-stride beneath a "
            "single harsh sodium streetlamp. On the wet concrete beneath his "
            "feet, just for this frame: a fractured broken version of the "
            "brand symbol from <<<image_2>>> — only a few of its four "
            "rotational geometric pieces are visible, glowing faintly cyan, "
            "as if the symbol is trying to assemble itself but failing. The "
            "rest of his footprint is empty. He casts NO normal shadow — only "
            "those few fragmented logo pieces. He looks down, alarmed. Use "
            "the exact geometry of the symbol from [Image2]."
        ),
        use_logo=True,
    ),
    Panel(
        idx=14,
        title="Confused",
        caption=None,
        body=(
            f"Close shot. {AGENT_DESC} kneels on the wet pavement, one palm "
            "down on the concrete where the fractured pieces appeared a moment "
            "ago. The pavement is now completely unmarked again. His cyan "
            "eye-points search the ground. Reflective puddle catches the "
            "streetlamp above. Quiet, intimate."
        ),
    ),
    Panel(
        idx=15,
        title="The Seam",
        caption="A thin glowing gradient seam ran through the pavement.",
        body=(
            f"Wide low-angle shot. {AGENT_DESC} stands on the city pavement. "
            "A thin glowing line — a horizontal cyan-to-warm gradient seam — "
            "runs across the floor between his feet and recedes into the deep "
            "perspective vanishing point of the city street. The seam is "
            "subtle, almost hidden between paving stones. His head tilts, "
            "noticing it for the first time. Atmospheric urban perspective."
        ),
    ),
    Panel(
        idx=16,
        title="The Path",
        caption="The fractured pieces formed a path.",
        body=(
            f"Medium shot. {AGENT_DESC} walks along the gradient seam through "
            "the city. Inside the seam, the broken geometric pieces of the "
            "brand symbol from <<<image_2>>> are arranged in a sequence "
            "like footprints, each fragment a piece of the symbol's four "
            "rotational lobes, leading deeper into the frame. Use the exact "
            "geometry of the symbol from [Image2]. Behind him, the city is "
            "blurred and silent."
        ),
        use_logo=True,
    ),
    Panel(
        idx=17,
        title="The City Peels Open",
        caption="The pavement separated. The buildings stretched upward.",
        body=(
            f"Wide cinematic shot. The city is mid-transformation: its asphalt "
            "and pavement are peeling open along the gradient seam like a "
            "vast unfolding skin, revealing layers beneath. Sealed buildings "
            "stretch upward and become translucent, their interiors visible. "
            f"{AGENT_DESC} stands at the center of the seam, awe-struck. The "
            "transition has the silent grandeur of a Foundation-era reveal."
        ),
    ),
    Panel(
        idx=18,
        title="The Compute World",
        caption="Beneath the city was the real infrastructure.",
        body=(
            "Vast cinematic establishing shot. Below the city, a sublime "
            "subterranean landscape: tall GPU towers like glowing dark "
            "cathedrals; floating model vaults like hanging libraries with "
            "luminous pages; massive transparent glass chambers refracting "
            "cyan light (TEE enclaves); and rivers of cyan-to-warm gradient "
            f"light flowing between them like onchain rails. {AGENT_DESC} is a "
            "tiny silhouette on a high mezzanine, gazing across the open vault. "
            "Awe, immense scale, dawn-cool palette."
        ),
    ),
    Panel(
        idx=19,
        title="Walking Deeper",
        caption="Every movement created a trace. Every trace wanted to become proof.",
        body=(
            f"OVER-THE-SHOULDER TRACKING-FROM-BEHIND shot. We see "
            f"{AGENT_DESC} entirely FROM BEHIND, his back to the camera, "
            "walking AWAY from us along a glass walkway through the compute "
            "world. The camera follows behind him. We never see his face — "
            "only the back of his head and body silhouetted against the "
            "world ahead. Embedded in the floor and walls along his path, "
            "the brand symbol from <<<image_2>>> appears in fragments — "
            "more complete than before, cyan glowing lines reassembling as "
            "he advances. Use the exact geometry of the symbol from [Image2]."
        ),
        use_logo=True,
    ),
    Panel(
        idx=20,
        title="The Verification Engine",
        caption="A massive verification engine. A sun held inside a machine.",
        body=(
            "OVER-THE-SHOULDER REAR VIEW. Wide cinematic shot from STRICTLY "
            "BEHIND the agent. The agent is shown ONLY as a back-silhouette "
            "in the lower-foreground center: we see ONLY the back of his "
            f"head and shoulders and back of his body — NO face features, "
            "NO eye-glow visible from this angle, NO front of his torso, "
            "NO front-facing arms. He is positioned between the camera and "
            "the engine like a person watching a stage, with his back fully "
            "to the camera and his entire body facing AWAY from us toward "
            "the engine ahead. \n\n"
            "Ahead of him, in the middle distance: a colossal spherical "
            "verification engine like a captured small star, suspended in "
            "a cage of dark elegant architecture. At the front of the "
            "engine (facing the agent and away from the camera), a clear "
            "lens-shaped aperture is shaped exactly like the [Your Brand] "
            "symbol from [Image2] — a 4-fold rotational geometric mark — "
            "pouring brilliant cyan light forward toward the agent's back, "
            "illuminating his rear silhouette in cool cyan rim-light. The "
            "agent is dwarfed by the engine ahead of him. We see his BACK, "
            "not his face. Use the exact geometry of the symbol from "
            "[Image2] for the lens. Like a Hopper-style figure-from-behind "
            "facing-into-an-immense-light composition."
        ),
        use_logo=True,
    ),
    Panel(
        idx=21,
        title="Cast a Shadow",
        caption="It asked only one thing: CAST A SHADOW.",
        body=(
            f"Mid shot. {AGENT_DESC} stands in front of the verification "
            "engine, illuminated by its cyan beam. Floating in clean cinematic "
            "type in the air above him, the words 'CAST A SHADOW' glow in soft "
            "cyan. The engine is silent. The agent's face is upturned. "
            "Reverent, monumental, museum-light intensity."
        ),
    ),
    Panel(
        idx=22,
        title="Performance Fails",
        caption=None,
        body=(
            f"Triptych dynamic montage in a single panel. {AGENT_DESC} "
            "appears three times in the frame, each pose more theatrical than "
            "the last: shouting torrents of luminous tokens upward, generating "
            "a paper-storm of receipts, posing oversized and shiny like a hero "
            "statue. Behind each pose, the engine remains dark and still. "
            "Failure is visible in the engine's silence. Stylized comic-strip-"
            "but-still-CG composition."
        ),
    ),
    Panel(
        idx=23,
        title="He Stops",
        caption="He stopped trying to be believed.",
        body=(
            f"Close shot. {AGENT_DESC} lowers his arms. His shoulders drop. "
            "His cyan eye-points dim slightly. The token-storm and paper-"
            "stream from the previous panel drift down past him as a soft "
            "snow. Nothing else moves. Stillness, exhaustion, surrender. "
            "Single overhead key-light."
        ),
    ),
    Panel(
        idx=24,
        title="Stepping Into the Light",
        caption="He stepped into the verification light.",
        body=(
            f"Wide cinematic shot from behind. {AGENT_DESC} steps forward into "
            "the cyan beam pouring from the verification engine. His silhouette "
            "is small against the column of light. The beam is so bright that "
            "we can see only his outline. Slow, deliberate motion."
        ),
    ),
    Panel(
        idx=25,
        title="The Inside",
        caption="Light passed through him. For the first time, we saw what was inside.",
        body=(
            f"Macro close-up of {AGENT_DESC} in the cyan beam, seen from the "
            "side. His porcelain-white body is now translucent: visible inside "
            "him are glowing geometric structures — model paths like neural "
            "lattice, inference traces like luminous threads, memory glyphs, "
            "small attestation seals, tiny crystalline proofs. Not organs, "
            "not wires — computation made visible. Awe and clarity."
        ),
    ),
    Panel(
        idx=26,
        title="Shadow Forms",
        caption="The shadow is solid.",
        body=(
            f"SHADOW OVERRIDE: this panel shows the agent's shadow as a "
            f"complete solid logo. Wide cinematic shot. {AGENT_DESC} stands "
            "in the cyan beam. On the dark floor stretching behind him, his "
            "shadow is the COMPLETE brand symbol from <<<image_2>>> — "
            "fully assembled, fully solid, fully unbroken, the entire 4-fold "
            "rotational mark with all four rounded petal-lobes around its "
            "center dot rendered in pure cyan light as a single intact "
            "glowing shape on the floor. NO fragments, NO flying pieces, "
            "NO broken parts, NO missing sections — the symbol is whole. "
            "Use the exact geometry of the symbol from [Image2]. The cyan "
            "logo-shadow is the only mark on the floor — no other normal "
            "shadow. Solid, clean, complete, museum-light intensity."
        ),
        use_logo=True,
    ),
    Panel(
        idx=27,
        title="A Shadow That Is a Logo",
        caption="He had a shadow.",
        body=(
            f"SHADOW OVERRIDE: this panel shows the agent's complete shadow. "
            f"Hero shot. {AGENT_DESC} turns and looks back over his shoulder "
            "at the floor behind him. His shadow stretches on the floor — but "
            "his shadow is not dark. It is the brand symbol from "
            "[Image2] rendered as a transparent geometric shape filled with "
            "soft cyan light, perfectly assembled, complete. Use the exact "
            "geometry of the symbol from [Image2]. He stares at it with "
            "quiet recognition. Museum-light intensity. No other normal "
            "shadow on the floor — only the cyan-light logo shadow."
        ),
        use_logo=True,
    ),
    Panel(
        idx=28,
        title="The Network",
        caption="Many shadows. One network. [Your tagline goes here]",
        body=(
            "SHADOW OVERRIDE: this panel shows many agents, each with their "
            "logo-shaped cyan-light shadow. Vast aerial wide shot, near-"
            "orbital view of the open landscape. Thousands of small agent-"
            "figures scattered across a luminous plain at dawn, each one "
            "casting a transparent cyan-light shadow of the [Your Brand] "
            "symbol from [Image2]. The shadows connect into a vast network "
            "of lines and nodes; from this altitude, the connected shadows "
            "resolve into a single colossal version of the [Your Brand] "
            "symbol covering the entire world below. Use the exact geometry "
            "of the symbol from [Image2]. Awe-inducing, symphonic, the "
            "structure of the world made visible."
        ),
        use_logo=True,
    ),
]


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    COMIC_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("shadow_comic")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--image-model", default=IMAGE_MODEL)
    p.add_argument("--bible-only", action="store_true", help="Generate the character bible and stop.")
    p.add_argument(
        "--only",
        nargs="*",
        type=int,
        help="Generate only these panel indexes (1-based). Useful for retries.",
    )
    return p.parse_args()


def load_replicate_client() -> Any:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise SystemExit("REPLICATE_API_TOKEN is required unless --dry-run.")
    import replicate

    return replicate.Client(api_token=token)


def bible_payload(fields: set[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": BIBLE_PROMPT,
        "aspect_ratio": "3:2",
        "quality": "high",
        "output_format": "png",
    }
    if fields:
        payload.update({"num_images": 1, "num_outputs": 1, "number_of_images": 1})
    return base.keep_supported_fields(payload, fields)


def panel_payload(panel: Panel, fields: set[str]) -> dict[str, Any]:
    full_prompt = f"{STYLE_HEADER}\n\n{panel.body}"
    refs: list[Path] = [BIBLE_PATH]
    if panel.use_logo:
        refs.append(LOGO_CYAN)

    payload: dict[str, Any] = {
        "prompt": full_prompt,
        "aspect_ratio": "3:2",
        "quality": "high",
        "output_format": "png",
        "input_images": refs,
    }
    if fields:
        payload.update({"num_images": 1, "num_outputs": 1, "number_of_images": 1})
    return base.keep_supported_fields(payload, fields)


def run_one_panel(
    client: Any,
    panel: Panel,
    fields: set[str],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, Any]:
    if panel.path.exists() and not args.force:
        logger.info("CACHE %s", panel.path.relative_to(ROOT))
        result = base.cached_result(panel.path, panel.body, args.image_model, ROOT)
    else:
        payload = panel_payload(panel, fields)
        result = base.run_replicate(client, args.image_model, payload, panel.path, logger, args.dry_run
        , ROOT)
    result.update({
        "id": panel.panel_id,
        "title": panel.title,
        "caption": panel.caption,
        "use_logo": panel.use_logo,
    })
    return result


def write_manifest(
    bible_result: dict[str, Any],
    panels: list[dict[str, Any]],
    image_model: str,
) -> None:
    all_results = [bible_result] + panels if bible_result else panels
    if not all_results:
        status = "empty"
    elif all(r["status"] == "dry_run" for r in all_results):
        status = "dry_run"
    elif all(r["status"] in {"ok", "cached"} for r in all_results):
        status = "ok"
    else:
        status = "partial"
    manifest = {
        "story_id": "shadow_comic",
        "title": "[Your Brand] — Cast a Shadow",
        "thesis": (
            "An AI agent has no shadow. The world refuses to recognize him "
            "until he stops performing and steps into a verification light. "
            "His shadow is the brand symbol. Other agents follow."
        ),
        "tagline": "[Your tagline goes here]",
        "status": status,
        "image_model": image_model,
        "created_by": "generate_shadow_comic.py",
        "updated_at": int(time.time()),
        "style_header": STYLE_HEADER,
        "agent_description": AGENT_DESC,
        "logo_path": str(LOGO_CYAN.relative_to(ROOT)),
        "bible": bible_result,
        "panels": panels,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = setup_logging()
    logger.info("shadow comic generation started workers=%d", args.workers)

    if not LOGO_CYAN.exists() and not args.dry_run:
        raise SystemExit(
            f"Missing {LOGO_CYAN.relative_to(ROOT)} — drop your brand logo PNG there.\n"
            f"See {LOGO_CYAN.parent.relative_to(ROOT)}/README.md for specs."
        )

    client = None if args.dry_run else load_replicate_client()
    image_fields = set() if args.dry_run else base.schema_fields(client, args.image_model, logger)

    # STEP 0 — character bible. Always required before scene panels.
    if BIBLE_PATH.exists() and not args.force:
        logger.info("CACHE %s", BIBLE_PATH.relative_to(ROOT))
        bible_result = base.cached_result(BIBLE_PATH, BIBLE_PROMPT, args.image_model, ROOT)
    else:
        payload = bible_payload(image_fields)
        bible_result = base.run_replicate(client, args.image_model, payload, BIBLE_PATH, logger, args.dry_run
        , ROOT)
    bible_result.update({"id": "agent_reference_sheet", "role": "character_bible"})

    if args.bible_only:
        write_manifest(bible_result, [], args.image_model)
        return 0 if bible_result["status"] in {"ok", "cached", "dry_run"} else 1

    if not (args.dry_run or BIBLE_PATH.exists()):
        logger.error("bible failed; cannot run scene panels")
        write_manifest(bible_result, [], args.image_model)
        return 1

    # STEP 1 — scene panels in parallel.
    only = set(args.only) if args.only else None
    todo = [p for p in PANELS if (only is None or p.idx in only)]
    if not todo:
        logger.error("no panels selected (check --only)")
        return 1

    panel_results: list[dict[str, Any]] = []
    if args.dry_run:
        for panel in todo:
            panel_results.append(run_one_panel(client, panel, image_fields, args, logger))
    else:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="panel") as pool:
            futures = {
                pool.submit(run_one_panel, client, panel, image_fields, args, logger): panel
                for panel in todo
            }
            for fut in as_completed(futures):
                panel = futures[fut]
                try:
                    panel_results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    logger.exception("FAIL panel %s: %s", panel.panel_id, exc)
                    panel_results.append({
                        "status": "error",
                        "id": panel.panel_id,
                        "title": panel.title,
                        "error": str(exc)[:800],
                    })

    panel_results.sort(key=lambda r: int(r.get("id", "99")))
    write_manifest(bible_result, panel_results, args.image_model)

    statuses = [r.get("status") for r in [bible_result] + panel_results]
    ok = {"ok", "cached", "dry_run"}
    if any(s not in ok for s in statuses):
        logger.error("one or more assets failed; see manifest for details")
        return 1
    logger.info("manifest written: %s", MANIFEST_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
