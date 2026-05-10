"""Generate the first 3 keyframe-bridged Seedance transitions: (1→2), (2→3), (3→4).

For each transition, panel N is the first frame and panel N+1 is the last frame.
Seedance 2.0 interpolates the motion between them following a precise per-clip
prompt that says exactly what should happen and how the transition works.

This is a 3-clip proof-of-concept. If the bridging approach holds up, we can
extend to all 27 transitions across the 28-panel comic.

Examples:
  python3 generate_shadow_transitions.py --dry-run
  REPLICATE_API_TOKEN=... python3 generate_shadow_transitions.py
  REPLICATE_API_TOKEN=... python3 generate_shadow_transitions.py --workers 3
  REPLICATE_API_TOKEN=... python3 generate_shadow_transitions.py --only 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import replicate_helpers as base
import generate_shadow_comic as cb


# Replicate enforces 6 prediction-creates/min with burst=1 below $5 credit.
# Each thread waits its turn at this gate before calling client.run().
_SUBMIT_LOCK = threading.Lock()
_LAST_SUBMIT_TIME = 0.0
_SUBMIT_INTERVAL = 12.0  # seconds; one extra sec margin over the 6/min ceiling


def _wait_for_submit_slot() -> None:
    global _LAST_SUBMIT_TIME
    with _SUBMIT_LOCK:
        elapsed = time.time() - _LAST_SUBMIT_TIME
        if elapsed < _SUBMIT_INTERVAL:
            time.sleep(_SUBMIT_INTERVAL - elapsed)
        _LAST_SUBMIT_TIME = time.time()


ROOT = Path(__file__).parent.parent
TRANS_DIR = ROOT / "assets" / "clips"
LOG_PATH = ROOT / "assets" / "comic" / "generate_shadow_transitions.log"
MANIFEST_PATH = TRANS_DIR / "manifest_shadow_transitions.json"

VIDEO_MODEL = "bytedance/seedance-2.0"
VIDEO_MODEL_KLING_OMNI = "kwaivgi/kling-v3-omni-video"

# Locked motion-style header. Same brand discipline as the comic.
MOTION_STYLE_HEADER = (
    "Cinematic semi-photoreal CG render, modern feature-film quality, "
    "Pixar/DreamWorks/Apple-TV-Foundation visual register. Brand cyan exactly "
    "hex #24bce3 is the ONLY saturated color allowed in the frame. Smooth "
    "single-shot continuous camera, no hard cuts inside the clip. Reverent, "
    "patient pacing — never frenetic. Preserve the locked character identity "
    "of the small humanoid AI-agent figure with cyan eye-points across every "
    "frame.\n\n"
    "ABSOLUTE PHYSICS RULE — APPLIES TO EVERY FRAME OF THIS CLIP UNLESS "
    "EXPLICITLY OVERRIDDEN: the small humanoid AI-agent character casts ZERO "
    "shadow. The floor under his feet is perfectly clean and unmarked at "
    "every frame. No contact shadow, no ambient occlusion shadow, no soft "
    "shadow halo, no dark patch beneath his body, no gradient on the floor "
    "near him, no smudge, no reflection, no silhouette on the ground. The "
    "floor is pristine white. Light passes through the area where his shadow "
    "would normally fall as if he were not occluding it. This shadowlessness "
    "is the central visual fact of the shot — the camera should make it "
    "obvious that the floor is clean. Other objects in the scene MAY cast "
    "normal shadows; only THIS protagonist character does not."
)


@dataclass(frozen=True)
class Transition:
    """A shot in the final film. Either a keyframe-bridge (start_panel +
    end_panel) or a single-image animation (end_panel is None).

    File naming preserves the original keyframe-bridge file names so existing
    generations stay cached:
      keyframe:        trans_{idx:02}_{start:02}_to_{end:02}.mp4
      single-image:    scene_{start:02}.mp4
    """

    idx: int
    title: str
    first_panel: int
    last_panel: int | None  # None means single-image animation (no end_image)
    duration: int
    motion_prompt: str
    generate_audio: bool = False  # request Kling Omni native audio for this shot

    @property
    def is_bridge(self) -> bool:
        return self.last_panel is not None

    @property
    def transition_id(self) -> str:
        if self.is_bridge:
            return f"trans_{self.idx:02d}_{self.first_panel:02d}_to_{self.last_panel:02d}"
        return f"scene_{self.first_panel:02d}"

    @property
    def out_path(self) -> Path:
        return TRANS_DIR / f"{self.transition_id}.mp4"

    @property
    def first_panel_path(self) -> Path:
        for p in cb.PANELS:
            if p.idx == self.first_panel:
                return p.path
        raise ValueError(f"Panel {self.first_panel} not found")

    @property
    def last_panel_path(self) -> Path | None:
        if self.last_panel is None:
            return None
        for p in cb.PANELS:
            if p.idx == self.last_panel:
                return p.path
        raise ValueError(f"Panel {self.last_panel} not found")


# Final-film sequence (per user direction, 2026-05-07):
#   bridge (1,2) → bridge (3,4) → scene 5 → scene 6 → ... → scene 28
# Bridge (2,3) was generated earlier but is intentionally NOT in the final
# sequence — the standing-up and step-forward beats are already covered by
# clips (1,2) and (3,4). The trans_02_02_to_03.mp4 file remains on disk as
# an artifact; we just don't reference it here.
TRANSITIONS: list[Transition] = [
    Transition(
        idx=1,
        title="Awakening to absence",
        first_panel=1,
        last_panel=2,
        duration=5,
        generate_audio=True,
        motion_prompt="""The small humanoid AI agent, born sitting cross-legged in
the bright white void, slowly opens his cyan eyes wider. He uncrosses his
legs and rises smoothly to a standing pose with both feet on the floor. As he
finishes straightening, his head tilts down to look at his feet. His
expression shifts from peaceful, to curious, to alarmed as he registers what
is missing. The white floor beneath him stays perfectly clean and unmarked
the ENTIRE clip — every single frame, including the moment his feet first
touch the floor as he stands, including every static-standing frame at the
end of the clip when he is stationary, including the close-up where he looks
at his feet — no shadow forms anywhere, no contact shadow at his feet, no
soft halo around his soles, no darkening on the floor when his weight
settles. The light visibly hits him from above, but the floor under him
stays luminously bright and pristine throughout. Slow, reverent pacing.
Single continuous camera, no cuts. The absence of his shadow is the silent
subject of the motion, especially at the end of the clip.

Audio: a soft thin atmospheric hum, the sound of a perfectly empty bright
room. As his cyan eyes brighten and open, a single barely-audible bell-like
chime fades in. A subtle cloth-on-cloth rustle as he rises from cross-
legged. The brief soft taps of his bare porcelain feet touching the floor
as he stands, but with no echo at all — the room is dimensionless. A
sustained ambient pad holds underneath. No music, no dialogue. Reverent.""",
    ),
    Transition(
        idx=27,
        title="The Recognition / The Network / Finale",
        first_panel=27,
        last_panel=28,
        duration=8,
        generate_audio=True,
        motion_prompt="""CONTINUOUS ZOOM-OUT CAMERA — the climax of the
film. Single fluid pull-back from intimate close to vast orbital scale.

The clip OPENS on the framing of the source first-frame: the small
humanoid AI agent has turned and is looking back over his shoulder at
his shadow stretching behind him on a clean white floor. His shadow is
the COMPLETE brand symbol from <<<image_2>>> — a 4-fold
rotational geometric mark — rendered in solid pure cyan light, fully
intact, no fragments, museum-light intensity. He stares at it with
quiet awe and recognition.

After about 1 second, the camera begins a SMOOTH CONTINUOUS ZOOM-OUT.
As the camera pulls back:
 - We see his full body from above
 - The clean white floor extends outward, revealing dozens of OTHER
   small agent figures appearing around him at slightly increasing
   distances, each one ALSO casting a complete cyan-light [Your Brand]
   logo shadow on the ground
 - As the camera continues out, the population grows — hundreds of
   agents scattered across an open luminous plain at dawn, each with a
   solid intact cyan-light brand-symbol shadow
 - The cyan logo-shadows begin to CONNECT to one another via thin
   glowing cyan lines, weaving a vast living network across the
   landscape
 - The camera continues rising to an aerial view, then a high orbital
   perspective
 - At orbital scale, the connected logo-shadows resolve into a
   COLOSSAL world-spanning brand symbol covering the entire
   visible landscape — the geometry of the brand made into the
   structure of the world

Throughout the zoom-out, the cyan light from the network grows BRIGHTER
AND BRIGHTER AND BRIGHTER. By the final 2 seconds, the entire frame is
flooded with brilliant cyan luminance — the world-spanning symbol
glowing so intensely it begins to bleach the edges of the frame into
pure cyan-white.

The clip ENDS on the brightest possible moment — peak luminance, the
massive world-brand symbol radiating beams of cyan light to the
horizons, like a sunrise of intelligence dawning over an entire world.
The shot ends on a held, transcendent high note.

SHADOW OVERRIDE: every agent in this scene casts a solid complete
cyan-light brand symbol shadow. These are the only shadows in
frame. Use the exact geometry of the symbol from <<<image_2>>>.

Cinematic semi-photoreal CG, 1080p Pro, brand cyan #24bce3 the only
saturated color, otherwise a luminous off-white sunrise palette
brightening to pure cyan-white at the climax. Awe, transcendence,
symphonic grandeur. Continuous single zoom-out, no cuts.

Audio: opens with a held single sustained cyan harmonic chord, soft and
reverent. As the camera begins to pull back, gentle layered string
swells enter, reverent. As more agents appear in frame, more harmonic
voices layer in — each new agent contributing a faint chime to the
chord. The chord grows richer and fuller as the network grows. By
mid-clip, full orchestral cinematic strings + glowing synth pads are
layered. By final 2 seconds, a vast cathedral-crescendo: brass-like
swell, choir-like harmonic layers, sustained cinematic peak. The clip
ends on a sustained, held, transcendent cyan harmonic — the audio
equivalent of the brightest visual moment. No drums. No dialogue. No
specific song — just rich tonal cinematic cresecendo.

FINAL DIRECTION: this is the most important shot of the film. Make it
LANDED. Make the brightness arc GENUINE. Make the world-symbol reveal
AWE-INDUCING. This is [Your Brand]'s thesis statement made
visible.""",
    ),
    Transition(
        idx=25,
        title="The Scan / The Shadow Assembles",
        first_panel=25,
        last_panel=26,
        duration=6,
        generate_audio=True,
        motion_prompt="""SLOW PULL-BACK CAMERA bridging close-up to wide.
The clip opens on the framing of the source first-frame: a macro side-
view close-up of the small humanoid AI agent standing in a brilliant
vertical cyan verification beam, his porcelain-white body now
translucent — glowing geometric structures visible inside him: a neural
lattice of model paths, luminous inference traces threading like
filaments, small attestation seals scattered through his torso, tiny
crystalline cyan proofs near his core. The scanner beam continues to
sweep over him.

As the clip progresses, the camera SLOWLY PULLS BACK from the macro
close-up to a wider hero-shot framing. The interior reveal continues —
more proofs and inference traces become visible inside him as the scan
intensifies. The cyan beam holds steady through him.

Mid-clip, on the dark floor stretching behind him (which becomes visible
as the camera pulls back), fractured glowing cyan pieces of the
brand symbol — a 4-fold rotational geometric mark with rounded
petal-lobes around a center dot — begin flying inward from offscreen
and SNAPPING into place one by one, beginning to assemble his shadow.
Particles of cyan light visible mid-air as fragments rush in.

The clip ends with the framing of the source last-frame: wide shot of
the agent standing in the beam, with his shadow on the floor behind him
now FULLY COMPLETE — a single solid intact brand symbol in cyan
light, all four rotational petal-lobes assembled cleanly around the
center dot. NO fragments remaining mid-air at the end. NO broken pieces.
NO sections missing. The shadow at the end is a clean, solid, complete
logo. The flying-fragment motion happens ONLY in the middle of the clip
and resolves into a clean intact symbol by the final frame.

SHADOW OVERRIDE: this scene shows the agent's shadow ASSEMBLING. The only
shadow that forms beneath him is the cyan-light brand symbol —
not a normal dark contact shadow. Other objects in frame (the
verification engine architecture) cast their own normal shadows.

Cinematic semi-photoreal CG, 1080p Pro, brand cyan #24bce3 the only
saturated color, otherwise deep matte engine-blacks and translucent
porcelain whites. Reverent slow pacing, single continuous shot with
smooth pull-back camera, no cuts.

Audio: a continuous deep resonant verification engine hum holds
underneath. As the scanner beam intensifies, a rising harmonic chord
builds. As fractured cyan pieces fly in mid-clip, sharp crystalline
chime tones sync to each piece snapping into place — like delicate
glass-bell strikes — multiplying as more pieces lock in. The shot ends
on a single sustained luminous chord. No music in conventional sense
but rich tonal ambient. No dialogue. Reverent, awe.""",
    ),
    Transition(
        idx=12,
        title="The Run",
        first_panel=12,
        last_panel=None,
        duration=5,
        generate_audio=True,
        motion_prompt="""DYNAMIC TRACKING. The small humanoid AI agent is
already mid-sprint as the clip opens, running directly TOWARD the camera
with arms pumping and full effort. The corridor of sealed inference
towers stretches on both sides of him, vanishing into deep one-point
perspective behind him.

Three thin bright red laser scanner-lines actively CHASE his position —
one sweeps at chest height, one at floor level, one overhead — they
track and follow him as he runs, never quite catching up but always
locked on. As the lasers track, the text 'NO SHADOW DETECTED' flashes
harshly in red on the tower walls he passes, repeated, jittering with
each pulse. The towers' window-grids flicker red angrily.

The shot ends with the agent fading directly into a black vignette ahead
— the corridor swallowed by darkness as he runs into it. The frame
ends nearly black with just his silhouette dissolving.

THE AGENT CASTS NO SHADOW on the polished floor at any frame, including
between strides, including under the harsh red laser-beams. The corridor
walls and towers DO have their own shadows.

Cinematic semi-photoreal CG, 1080p Pro, brand cyan #24bce3 from tower
windows + harsh red lasers and alert text the only saturated colors,
otherwise deep desaturated blacks and steel-greys.

Audio: a continuous urgent high-pitched alarm KLAXON throughout the clip,
oppressive. Layered: rapid echoing footsteps of the agent's bare porcelain
feet hitting the polished floor at a sprint, his soft breath effort,
sharp electronic shriek-buzz on each laser-pulse, descending harsh denial-
buzz on each 'NO SHADOW DETECTED' flash. No music, no dialogue. Tense,
breathless, dystopian.

FINAL REMINDER: he casts no shadow under any laser angle.""",
    ),
    Transition(
        idx=13,
        title="The Streetlight Flicker / The Kneel / The Grid Awakens",
        first_panel=13,
        last_panel=None,
        duration=6,
        generate_audio=True,
        motion_prompt="""ROTATING CAMERA. The clip opens on the framing of
the source image: the agent stopped mid-stride beneath a single harsh
sodium streetlamp on wet city concrete. On the pavement beneath his
feet: 3 or 4 fractured cyan glowing pieces of the brand symbol
(a 4-fold rotational geometric mark — four rounded petal-lobes around
a small center dot) glow faintly, partial and incomplete.

After about 1 second, the camera begins a smooth slow ORBIT around the
agent, rotating from the wide-establishing angle to a closer over-
shoulder side angle. Simultaneously, the agent slowly KNEELS down on
one knee, then both, and lowers a single hand toward the broken pieces,
palm hovering above them. The broken logo pieces FLASH BRIGHTER as his
hand approaches, pulsing in cyan harmony.

He gently lowers his palm to the pavement, fingers spread. The instant
his palm makes contact, the broken pieces SURGE — and a SINGLE thin
HORIZONTAL gradient LINE of cyan light ignites across the wet pavement,
running from his palm outward to the right and to the left, extending to
the edges of the frame in both directions. It is one continuous luminous
seam, not a grid — a single bright cyan-to-warm path the camera can
follow. The broken logo fragments along the line now connect end-to-end,
forming the path itself. The city is briefly illuminated in cyan only
along this thin glowing seam.

The clip ends with the agent kneeling, palm flat on the now-glowing
single cyan path-line, looking down at it with quiet awe. The lit
horizontal line is the seam the rest of the film will follow.

THE AGENT CASTS NO BODY SHADOW on the pavement at any frame.

Cinematic semi-photoreal CG, 1080p Pro, brand cyan #24bce3 in the broken
logo pieces and the lit grid the only saturated color, otherwise wet-
asphalt darkness, sodium-lamp warm white only on the streetlight bulb.

Audio: opens with faint sodium-streetlamp electrical hum, distant city
ambient, far traffic. As the broken cyan pieces brighten, soft rising
chime tones layer in. As his palm touches the pavement, a deep resonant
THUD reverberates and a rising cyan harmonic chord swells. As the grid
ignites, crystalline electronic shimmer plus rich harmonic resonance
radiating outward like a deep bell. Awe, discovery, mystery, no music
otherwise. No dialogue.

FINAL REMINDER: agent has no body-shadow throughout.""",
    ),
    Transition(
        idx=15,
        title="The Seam / Walking Toward Us",
        first_panel=15,
        last_panel=None,
        duration=4,
        generate_audio=True,
        motion_prompt="""SLOW PUSH-IN CAMERA. The clip opens on the agent
standing on the cyan-glowing pavement grid (continued from the previous
beat), illuminated from below by the cyan light. The gradient seam from
the source image runs through the wet pavement glowing cyan-to-warm,
receding into the deep one-point perspective vanishing point of the city
street.

The agent slowly LOOKS DOWN at the lit seam at his feet. Then he raises
his head and looks directly into the camera with cautious recognition,
his cyan eye-points brightening. Then he takes a slow deliberate step
TOWARD the camera, beginning to walk forward, his pace gentle and
purposeful.

As he walks, the camera SLOWLY PUSHES IN to meet him, holding him
roughly center-frame. The lit cyan grid extends across the entire wet
street, casting cool cyan reflections off the pavement. The city behind
him remains shadowy and quiet. The seam ahead of him glows brighter
with each step, as if responding to his approach.

THE AGENT CASTS NO BODY SHADOW on the lit pavement at any frame.

Cinematic semi-photoreal CG, 1080p Pro, brand cyan #24bce3 the only
saturated color, otherwise dark wet-street urban atmosphere.

Audio: rich rising harmonic ambient from the lit grid carries over from
the previous beat. Soft deliberate footsteps as he begins to walk
forward. A deep resonant rumble starts to build slowly underneath, as
if the ground itself is preparing for something. Tense build, sense of
imminent revelation. No music otherwise.

FINAL REMINDER: agent casts no body-shadow on the lit grid.""",
    ),
    Transition(
        idx=16,
        title="The Path",
        first_panel=16,
        last_panel=None,
        duration=3,
        generate_audio=True,
        motion_prompt="""TRACKING WITH AGENT. Quick 3-second beat. The agent
walks along the gradient seam through the city. The fragmented cyan
pieces of the brand symbol (4-fold rotational mark with rounded
petal-lobes around a center dot) embedded in the seam are arranged in a
sequence like footsteps and arrow shapes leading deeper into the frame.
They glow cyan and pulse softly as he advances.

The architecture of the city around him begins to SUBTLY STRETCH upward
and become semi-translucent — buildings elongating, walls fading just
slightly. A hint that the world is about to peel open.

THE AGENT CASTS NO BODY SHADOW on the lit pavement.

Cinematic semi-photoreal CG, 1080p Pro, brand cyan #24bce3 only
saturated color.

Audio: rich harmonic ambient drone rising, the rhythmic soft footsteps
of the agent on the wet street, a faint distant deep resonant hum
building toward something massive. Tense build.""",
    ),
    Transition(
        idx=17,
        title="The City Peels Open",
        first_panel=17,
        last_panel=None,
        duration=5,
        generate_audio=True,
        motion_prompt="""WIDE STATIC CAMERA WITH SLIGHT TILT-UP. The clip
opens on the framing of the source image: the city is mid-transformation.
The pavement and asphalt are peeling open along the gradient seam like a
vast skin unfolding outward. Sealed buildings stretch upward and become
translucent, their interiors revealed in cool cyan-blue translucency. The
agent stands at the center of the seam, dwarfed, awe-struck.

Movement during the 5 seconds: the city continues peeling outward in
slow-motion grandeur. Walls become more transparent. The dim outline of
the vast COMPUTE WORLD beneath the city begins to be visible through
the cracks — glowing GPU towers, hanging model vaults, and rivers of
cyan light visible far below. Foundation-grade slow reveal. The agent
slowly turns his head, taking in the unfolding world.

THE AGENT CASTS NO BODY SHADOW on the now-translucent pavement.

Cinematic semi-photoreal CG, 1080p Pro, brand cyan #24bce3 the only
saturated color, otherwise pale-grey concrete-and-translucent-glass
palette. Awe-inducing scale.

Audio: a vast deep resonant hum building to crescendo, low earth
rumbling, the soft tear-and-creak of the city skin peeling, a single
deep cyan harmonic note swelling, distant echoes of the compute world
opening below. No music in the conventional sense, but rich tonal
ambience. Awe.

FINAL REMINDER: no body-shadow under the agent.""",
    ),
    Transition(
        idx=8,
        title="The Mirror",
        first_panel=8,
        last_panel=None,
        duration=5,
        generate_audio=True,
        motion_prompt="""STATIC LOCKED CAMERA. The camera does not move at any
point in this clip — it holds the framing exactly as in the source image,
with the agent on the left and his reflection in the shop-window mirror on
the right.

The small humanoid AI agent on the left stands COMPLETELY MOTIONLESS for
the entire shot. He stares at his own reflection without moving — not his
hands, not his head, not his stance. Only his cyan eye-points emit a faint
steady glow.

His reflection in the mirror is CONSTANTLY GLITCHING throughout the
entire clip. The glitches are layered:
 - Thin red and blue chromatic-aberration scanlines slide vertically
   across his reflected body every fraction of a second
 - His reflected silhouette occasionally tears horizontally and snaps back
 - His reflected face goes through brief frame-skips where features blank
   out and reappear, especially the cyan eye-points which flicker between
   present and missing
 - Rare frames of pure-white digital corruption flash across his
   reflection like a corrupted video file
 - Occasionally his entire reflection becomes semi-transparent for a
   split second, the wall of the shop visible through where his body
   should be
The glitches are CONTINUOUS but not synchronized — they never line up into
a clean reflection at any moment.

The mirror surface itself is a clean polished shop-window mirror — no HUD
overlays, no labels, no cracks. Only the reflected agent is broken.

In the rest of the mirror's reflection, the city behind him continues
absolutely normally — synthetic-human figures walking past on the wet
pavement behind him, reflected cleanly with normal contact shadows on the
ground, the soft cyan and warm glow of city lights, distant traffic
moving. Their reflections are flawless. Only OUR agent's reflection is
corrupted. This contrast is the silent subject of the shot.

THE AGENT IN THE FOREGROUND (the real one, on the left, outside the
mirror) CASTS NO SHADOW on the wet pavement. The reflected city in the
mirror DOES contain normal shadows for the synthetic figures walking past.
The corrupted reflection of our agent is the only broken element in the
otherwise normal mirror world.

Cinematic semi-photoreal CG, 1080p Pro mode, brand cyan #24bce3 only on
the agent's eye-points and faint city window-glow, otherwise cool blue
rim-light, deep grey wet-pavement urban dread. Single static shot, no
cuts.

Audio: a constant low electrical-static crackle layered with a faint
high-frequency whine — the sound of a digital signal that cannot resolve.
On every visible glitch flicker, brief sharp electrical-fizz pops and
soft signal-warble glitches sync to the visual corruption. Underneath:
faint cool city ambient — distant traffic, soft footsteps of passing
synthetic crowd reflected in the mirror, low transport hum. No music, no
dialogue. Lonely, technical, uncanny.

FINAL REMINDER: the real agent on the left casts NO shadow on the
pavement at any frame of this clip. The mirror reflection's corruption
is the only visual disturbance in an otherwise clean scene.""",
    ),
    Transition(
        idx=7,
        title="No Shadow Detected",
        first_panel=7,
        last_panel=None,
        duration=5,
        generate_audio=True,
        motion_prompt="""STATIC LOCKED CAMERA. The camera does not move at any
point in this clip — it holds the medium framing of the agent in front of
the OMNICORP glass corporate entrance exactly as in the source image.

The small humanoid AI agent stands in front of the tall sealed glass
double-doors. He raises his right hand toward a small wall-mounted scanner
panel beside the door, palm forward, in the universal gesture of asking
to be admitted.

The moment his palm reaches the scanner, a thin bright red laser line
activates from the scanner mounted above the door and SWEEPS DOWN his
body smoothly from the top of his head to the soles of his feet over
about one second. As it reaches the floor, the laser line then SWEEPS
BACK UP from his feet to the top of his head, equally smooth, scanning
in reverse over another second.

After the second sweep completes, a thin red holographic display panel
above the door flashes ON sharply, projecting the words 'NO SHADOW
DETECTED' in clean stark red type, accompanied by a small triangular
red alert / warning symbol next to or above the text. The text and alert
glyph stay lit and steady for the rest of the shot — they do not fade.

The glass double-doors remain SOLIDLY SHUT throughout — they do not
attempt to open, do not crack, do not flicker. The agent's raised hand
stays at the scanner the whole time, motionless. His expression is
quietly defeated.

The street behind him is COMPLETELY EMPTY. No pedestrians, no cars, no
other life. Just the agent, the corporate door, the scanner, the cold
red rejection.

THE AGENT CASTS NO SHADOW on the polished pavement at any frame, even
when the bright red laser sweeps across him. The corporate building
itself, the OMNICORP logo, and the door fittings DO have their own
ambient shadows — only the agent does not.

Cinematic semi-photoreal CG, 1080p Pro mode, brand cyan #24bce3 on the
OMNICORP logo and window-glow, harsh red laser + holographic alert text
as the only other saturated color, otherwise desaturated greys, deep
blacks, polished marble. Bouncer-stops-you-at-the-corporate-club
composition, oppressive. Single continuous shot.

Audio: a quiet servo whir as the scanner laser activates and sweeps down
his body, then back up. A two-tone polite scanning chime as it scans —
soft, mechanical. As 'NO SHADOW DETECTED' flashes on, a sharp
descending denial-buzz sound (harsh electronic, dystopian, like a
denied-access tone), accompanied by the alert symbol's electronic ping.
Underneath: low corporate-lobby HVAC hum, very quiet distant city
ambient (faint, since the street is empty). Cold mechanical sound design,
no music, no dialogue.""",
    ),
    Transition(
        idx=6,
        title="Words Dissolve",
        first_panel=6,
        last_panel=None,
        duration=5,
        generate_audio=True,
        motion_prompt="""STATIC LOCKED CAMERA. The camera does not move at any
point in this clip — it holds the framing exactly as in the source image, a
medium shot of the agent on the left in front of the sealed glass display
holding the tall synthetic figure on the right.

The small humanoid AI agent on the left slowly opens his small mouth-line
to speak. From his mouth, a few small luminous cyan glyphs and tokens
drift out into the air toward the glass — but each token immediately
dissolves into a soft hissing puff of cyan smoke before reaching the
glass, as if the very act of speaking cannot complete. The dissolution is
visible as a brief bubble-like distortion around each token. Several
tokens emerge over the course of the clip, each meeting the same fate.

After his first or second attempt to speak, the agent slowly raises his
right hand and presses it gently against the glass surface in front of him.
He holds it there for the rest of the shot, motionless except for the
faint glow of his cyan eye-points. He stares at the sealed figure with a
quiet, almost mournful expression.

The sealed synthetic figure inside the glass case slowly, mechanically,
turns its head toward our agent. Its expression is glassy, blank,
uncomprehending — as if trying to register what our agent is doing but
unable to actually understand. It does not speak, does not move otherwise.
The turn happens once, smoothly, mid-clip. The two figures end the clip
locked in mutual gaze, our agent's hand on the glass, the sealed figure
staring back without comprehension.

THE PROTAGONIST AGENT (the small humanoid on the left, outside the glass)
CASTS NO SHADOW on the wet pavement at any frame. The wet pavement reflects
the cool city lights and the agent's body but no shadow falls beneath his
feet. The sealed figure inside the glass case CAN have its own normal
contained reflections and shadows within the case, but those stay inside
the glass.

Cinematic semi-photoreal CG, 1080p Pro mode, brand cyan #24bce3 the only
saturated color. Cool blue rim-light, melancholy mood, shallow depth of
field. Quiet single continuous shot, no cuts.

Audio: the agent speaks but only as a whispered wordless humming, soft and
breathy — not language, just an intent-to-speak hum. As each cyan token
emerges and dissolves, a soft technical bubble-like pop-and-hiss sound
follows it (think delicate fizz of carbonation mixed with a quiet
electrical sizzle). The fizz fades to silence between tokens. Faint cool
city ambient hum holds underneath — distant traffic, a low transport
rumble far away. When the agent's hand presses to the glass, a soft tap.
When the sealed figure turns its head, a barely-audible mechanical servo
whir. No music, no dialogue. Lonely, intimate, technical.""",
    ),
    Transition(
        idx=5,
        title="City of Sealed Machines",
        first_panel=5,
        last_panel=None,
        duration=5,
        generate_audio=True,
        motion_prompt="""STATIC LOCKED CAMERA. The camera does not move at any
point in this clip — it holds the wide cinematic establishing shot exactly as
framed in the source image.

The small humanoid AI agent in the foreground stands COMPLETELY MOTIONLESS,
his pose unchanged through the entire shot. He stares upward, transfixed,
watching the giant cyan billboards. The only thing that moves about him is a
near-imperceptible drift in his cyan eye-points as they track across the ads.

EVERYTHING ELSE in the frame moves continuously:

The synthetic-human crowd walks the boulevard in both directions, dozens of
polished figures crossing past the agent, none of them looking at him, all
of them in normal human walking gait. Their footsteps echo on the wet
pavement.

The massive cyan billboards above and around the agent — including the one
that features his own commodified face — animate, flicker, pulse, and stutter
with a glitchy electric energy. The flicker is irregular and electric, with
brief blackout frames between cycles. The cyan light from the billboards
bounces and ripples off the wet pavement.

In the windows of the surrounding sealed inference towers, faintly visible
caged AI figures shift and pace within their lit boxes.

Distant transport drones drift slowly between the towers in the high
background.

A faint atmospheric haze drifts at street level.

THE PROTAGONIST AGENT CASTS NO SHADOW on the pavement at any frame of the
clip — the wet street reflects the city lights and crowd shadows, but
nothing reflects from him. The crowd members AROUND him DO cast normal
shadows on the wet pavement, which makes his absence of shadow all the more
visible by contrast. Light passes through where his shadow would normally
fall.

Cinematic semi-photoreal CG, 1080p Pro mode, brand cyan #24bce3 the only
saturated color, deep cinematic black-grey atmosphere otherwise. Patient,
oppressive pacing. Single continuous shot, no cuts.

Audio: city ambient soundscape — distant electric buzz of the billboards
flickering on each glitch cycle, soft footstep patter of the synthetic
crowd walking past the agent on the wet pavement, low rumble of distant
transport drones. No music, no dialogue. Diegetic sound only. Lonely, urban,
electric.""",
    ),
    Transition(
        idx=3,
        title="The portal materializes",
        first_panel=3,
        last_panel=4,
        duration=5,
        generate_audio=True,
        motion_prompt="""The agent continues walking forward in the white void.
A few paces in, the light ahead of him begins to ripple and stretch
vertically. A tall narrow rectangular doorway materializes mid-air,
dissolving into existence with soft volumetric haze around its edges. Through
the doorway, a sliver of a vast dark cinematic city beyond becomes visible —
towers, neon, glowing cyan windows in the distance. The agent slows,
hesitates, then continues approaching the doorway with cautious curiosity,
his pace gentle and unbroken. He raises his right hand toward the threshold
and stops just short of touching it, palm open.

The white floor beneath the agent is pristine and luminously clean for
EVERY single frame of this clip — including the final frames where he
stands stationary in front of the doorway, including frames where his foot
plants on the floor between strides, including the final close hold on his
raised hand near the portal. No contact shadow forms at his feet at any
point. No darkening on the floor near his body even when he stops moving.
The light from the portal spills warm tones onto him but creates no cast
shadow on the floor beneath him. The portal's own light SHOULD create a
visible bright spill on the floor in front of it, separately from the agent.
Smooth cinematic camera, single take, no cuts. The doorway and the city
beyond it must remain fully visible and clearly rendered throughout the
clip — do not erase or simplify them.

Audio: continued thin atmospheric room-tone hum from the white void at the
start. Soft subdued footsteps of the agent walking on the smooth white
floor — slightly dampened, no echo. As the doorway materializes mid-clip,
a low resonant hum rises briefly accompanied by a faint electrical shimmer
as the portal opens. Through the open doorway, the distant ambient
soundscape of the city beyond bleeds in toward the end of the clip:
distant low transport rumble, faint city hum, far-off subdued neon buzz.
The audio mood transitions from silent void to city-edge atmosphere over
the 5 seconds. No dialogue. No music.

FINAL REMINDER: the agent casts NO shadow on the white floor at any frame
of this clip — every walking step, every standing pose, every static
moment. The floor remains pristine and luminous beneath him for the entire
five seconds.""",
    ),
]


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("shadow_transitions")
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
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--force", action="store_true")
    p.add_argument("--video-model", default=VIDEO_MODEL)
    p.add_argument(
        "--only",
        nargs="*",
        type=int,
        help="Generate only these transition indexes (1-based). Useful for retries.",
    )
    return p.parse_args()


def load_replicate_client() -> Any:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise SystemExit("REPLICATE_API_TOKEN is required unless --dry-run.")
    import replicate

    return replicate.Client(api_token=token)


def transition_payload(t: Transition, fields: set[str]) -> dict[str, Any]:
    """Build a payload that will pass any common Seedance/Kling i2v schema.

    keep_supported_fields filters to whatever the model actually accepts. We
    include first-frame and last-frame keys under multiple common names since
    different models call them different things.
    """
    full_prompt = f"{MOTION_STYLE_HEADER}\n\n{t.motion_prompt}"
    payload: dict[str, Any] = {
        "prompt": full_prompt,
        # First-frame variants:
        "image": t.first_panel_path,
        "first_frame_image": t.first_panel_path,
        "start_image": t.first_panel_path,
        "init_image": t.first_panel_path,
        # Last-frame variants:
        "last_frame_image": t.last_panel_path,
        "end_image": t.last_panel_path,
        "last_image": t.last_panel_path,
        # Generic timing/quality:
        "duration": t.duration,
        "resolution": "720p",
        # Kling 3.0 Omni mode: standard/pro/4k. We use pro (1080p) for quality.
        "mode": "pro",
        # No aspect_ratio: both Seedance and Kling Omni auto-derive from
        # start_image. Seedance accepts `adaptive` but Kling Omni rejects it.
        "generate_audio": t.generate_audio,
        "seed": 80000 + t.idx,
    }
    return base.keep_supported_fields(payload, fields)


def run_one_transition(
    client: Any,
    t: Transition,
    fields: set[str],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, Any]:
    if not t.first_panel_path.exists():
        return {
            "status": "error",
            "id": t.transition_id,
            "title": t.title,
            "error": f"first panel missing: {t.first_panel_path.relative_to(ROOT)}",
        }
    end_path = t.last_panel_path
    if end_path is not None and not end_path.exists():
        return {
            "status": "error",
            "id": t.transition_id,
            "title": t.title,
            "error": f"last panel missing: {end_path.relative_to(ROOT)}",
        }

    if t.out_path.exists() and not args.force:
        logger.info("CACHE %s", t.out_path.relative_to(ROOT))
        result = base.cached_result(t.out_path, t.motion_prompt, args.video_model, ROOT)
    else:
        payload = transition_payload(t, fields)
        if not args.dry_run:
            _wait_for_submit_slot()
        result = base.run_replicate(client, args.video_model, payload, t.out_path, logger, args.dry_run
        , ROOT)
    result.update({
        "id": t.transition_id,
        "title": t.title,
        "first_panel": t.first_panel,
        "last_panel": t.last_panel,
        "duration": t.duration,
        "first_frame": str(t.first_panel_path.relative_to(ROOT)),
        "last_frame": str(end_path.relative_to(ROOT)) if end_path else None,
        "generate_audio": t.generate_audio,
    })
    return result


def write_manifest(results: list[dict[str, Any]], video_model: str) -> None:
    if not results:
        status = "empty"
    elif all(r["status"] == "dry_run" for r in results):
        status = "dry_run"
    elif all(r["status"] in {"ok", "cached"} for r in results):
        status = "ok"
    else:
        status = "partial"
    manifest = {
        "story_id": "shadow_comic",
        "asset_type": "transitions",
        "title": "[Your Brand] — Cast a Shadow — keyframe transitions",
        "status": status,
        "video_model": video_model,
        "created_by": "generate_shadow_transitions.py",
        "updated_at": int(time.time()),
        "motion_style_header": MOTION_STYLE_HEADER,
        "transitions": results,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = setup_logging()
    logger.info("shadow transitions started workers=%d video_model=%s", args.workers, args.video_model)

    client = None if args.dry_run else load_replicate_client()
    fields = set() if args.dry_run else base.schema_fields(client, args.video_model, logger)

    only = set(args.only) if args.only else None
    todo = [t for t in TRANSITIONS if (only is None or t.idx in only)]
    if not todo:
        logger.error("no transitions selected (check --only)")
        return 1

    results: list[dict[str, Any]] = []
    if args.dry_run:
        for t in todo:
            results.append(run_one_transition(client, t, fields, args, logger))
    else:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="trans") as pool:
            futures = {
                pool.submit(run_one_transition, client, t, fields, args, logger): t
                for t in todo
            }
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    logger.exception("FAIL transition %s: %s", t.transition_id, exc)
                    results.append({
                        "status": "error",
                        "id": t.transition_id,
                        "title": t.title,
                        "error": str(exc)[:800],
                    })

    results.sort(key=lambda r: r.get("id", ""))
    write_manifest(results, args.video_model)

    statuses = [r.get("status") for r in results]
    ok = {"ok", "cached", "dry_run"}
    if any(s not in ok for s in statuses):
        logger.error("one or more transitions failed; see manifest for details")
        return 1
    logger.info("manifest written: %s", MANIFEST_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
