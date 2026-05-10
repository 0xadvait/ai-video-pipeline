"""Stitch the 13 'Cast a Shadow' clips into the final film.

Final film order (per user direction across the session):
  1. trans_01_01_to_02      — Awakening to absence
  2. trans_03_03_to_04      — Portal materializes
  3. scene_05               — City of sealed machines
  4. scene_06               — Words dissolve
  5. multi_9_10_11          — Affliction triptych (hollow / luxury / abandoned)
  6. scene_08               — The mirror (user's trimmed cut)
  7. scene_07               — OMNICORP rejection
  8. scene_12               — The run
  9. scene_13               — Palm-touch ignites the line
 10. multi_15_16_17         — Continuous walk (seam → path → city peels)
 11. multi_18_19_20         — Compute world arc (back-of-agent)
 12. trans_25_25_to_26      — Scan + shadow assembles into solid logo
 13. trans_27_27_to_28      — Recognition + zoom out + finale

Output: assets/shadow_comic/cast_a_shadow_final.mp4 (native Kling audio)
        assets/shadow_comic/cast_a_shadow_final_with_music.mp4 (Cyan Proof
        layered under native audio at -8 dB)

Each clip's native Kling audio is preserved. The music bed is mixed in
underneath at low gain so the diegetic sounds (billboard buzz, footsteps,
denial-buzz, engine hum, finale crescendo) stay legible.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRANS_DIR = ROOT / "assets" / "clips"
OUT_DIR = ROOT / "assets"
OUT_NATIVE = OUT_DIR / "cast_a_shadow_final.mp4"
OUT_MUSIC = OUT_DIR / "cast_a_shadow_final_with_music.mp4"
CONCAT_LIST = TRANS_DIR / "cast_a_shadow_concat.txt"
MUSIC_PATH = ROOT / "assets" / "audio" / "music_bed.mp3"

CLIPS = [
    "trans_01_01_to_02.mp4",
    "trans_03_03_to_04.mp4",
    "scene_05.mp4",
    "scene_06.mp4",
    "multi_9_10_11.mp4",
    "scene_08.mp4",
    "scene_07_reversed.mp4",
    "scene_12.mp4",
    "scene_13.mp4",
    "multi_15_16_17.mp4",
    "multi_18_19_20.mp4",
    "trans_25_25_to_26.mp4",
    "trans_27_27_to_28.mp4",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-music", action="store_true", help="Skip the music-bed mix.")
    p.add_argument(
        "--music-gain-db",
        type=float,
        default=-2.0,
        help="Music bed gain in dB under the native audio (default -2).",
    )
    p.add_argument(
        "--music-start",
        type=float,
        default=0.0,
        help="Seconds into Cyan Proof to start (default 0).",
    )
    p.add_argument(
        "--fade-in",
        type=float,
        default=4.0,
        help="Music fade-in seconds (default 4).",
    )
    p.add_argument(
        "--fade-out",
        type=float,
        default=4.0,
        help="Music fade-out seconds (default 4).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite intermediates.")
    return p.parse_args()


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found on PATH.")


def assert_clips_present() -> list[Path]:
    paths = [TRANS_DIR / name for name in CLIPS]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing clips: " + ", ".join(missing))
    return paths


def write_concat_list(paths: list[Path]) -> None:
    lines = [f"file '{p.resolve()}'" for p in paths]
    CONCAT_LIST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_with_native_audio(paths: list[Path], out: Path) -> None:
    """Concat all clips using filter_complex concat — robust to stream-order
    differences across inputs.

    The `-f concat` demuxer silently drops video from inputs whose stream
    order doesn't match the first input; some Kling clips have audio first,
    others video first. filter_complex explicitly maps each stream and
    re-encodes to a uniform H.264/AAC baseline.
    """
    write_concat_list(paths)  # kept for human reference / debugging
    n = len(paths)
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", str(p)]
    # Build filter graph: scale every video to 1764x1176 24fps, every audio to
    # 48000Hz stereo, then concat n clips.
    filt_parts: list[str] = []
    concat_inputs: list[str] = []
    for i in range(n):
        filt_parts.append(
            f"[{i}:v:0]scale=1764:1176:force_original_aspect_ratio=decrease,"
            f"pad=1764:1176:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24[v{i}]"
        )
        filt_parts.append(
            f"[{i}:a:0]aresample=48000,aformat=channel_layouts=stereo[a{i}]"
        )
        concat_inputs.append(f"[v{i}][a{i}]")
    filt_parts.append(
        "".join(concat_inputs) + f"concat=n={n}:v=1:a=1[outv][outa]"
    )
    cmd += [
        "-filter_complex", ";".join(filt_parts),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def total_duration(out: Path) -> float:
    return float(
        subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(out)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )


def mix_music(native: Path, music: Path, args: argparse.Namespace, out: Path) -> None:
    """Mix Cyan Proof.mp3 under the native audio at -gain dB."""
    duration = total_duration(native)
    fade_out_start = max(0.0, duration - args.fade_out)
    music_filter = (
        f"atrim=start=0:duration={duration},"
        f"asetpts=PTS-STARTPTS,"
        f"volume={args.music_gain_db}dB,"
        f"afade=t=in:st=0:d={args.fade_in},"
        f"afade=t=out:st={fade_out_start}:d={args.fade_out}"
    )
    cmd = [
        "ffmpeg",
        "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(native),
        "-ss", str(args.music_start),
        "-i", str(music),
        "-t", str(duration),
        "-filter_complex",
        # `normalize=0` keeps each stream at its own gain (default amix would
        # halve each input). Native audio passes through unchanged; the
        # music bed sits underneath at the user-set gain.
        f"[1:a]{music_filter}[bed];[0:a][bed]amix=inputs=2:normalize=0:duration=first:dropout_transition=0[a]",
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    require_ffmpeg()
    paths = assert_clips_present()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Stitching {len(paths)} clips with native audio ===")
    if not OUT_NATIVE.exists() or args.force:
        concat_with_native_audio(paths, OUT_NATIVE)
    duration = total_duration(OUT_NATIVE)
    size = OUT_NATIVE.stat().st_size / 1_000_000
    print(f"  OK {OUT_NATIVE.relative_to(ROOT)} ({size:.1f} MB, {duration:.1f}s)")

    if args.no_music:
        return 0

    if not MUSIC_PATH.exists():
        print(f"WARN music not found at {MUSIC_PATH} — skipping music mix")
        return 0

    print(f"\n=== Mixing music bed ({MUSIC_PATH.name} at {args.music_gain_db} dB) ===")
    mix_music(OUT_NATIVE, MUSIC_PATH, args, OUT_MUSIC)
    msize = OUT_MUSIC.stat().st_size / 1_000_000
    print(f"  OK {OUT_MUSIC.relative_to(ROOT)} ({msize:.1f} MB, {duration:.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
