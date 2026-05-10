"""Compile the 28 generated panels into a comic-book grid PNG with captions.

Mirrors compile_blackbox_comic.py but reads PANELS from generate_shadow_comic.
Default layout 4 cols x 7 rows = 28 cells exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import generate_shadow_comic as cb


ROOT = Path(__file__).parent.parent
OUT_PATH = cb.COMIC_DIR / "shadow_comic.png"

BG = (4, 19, 23)              # deep-teal-black
PANEL_BG = (10, 15, 25)       # secondary navy
TEXT = (233, 248, 252)
ACCENT = (36, 188, 227)
MUTED = (136, 156, 168)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--rows", type=int, default=7)
    p.add_argument("--panel-width", type=int, default=900)
    p.add_argument("--margin", type=int, default=32)
    p.add_argument("--caption-h", type=int, default=130)
    return p.parse_args()


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    if not text:
        return []
    words, lines, current = text.split(), [], ""
    for w in words:
        trial = (current + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w:
            current = trial
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def draw_caption(canvas: Image.Image, panel: cb.Panel, x: int, y: int, w: int) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = find_font(24, bold=True)
    cap_font = find_font(20)
    num_font = find_font(18, bold=True)

    # Number badge.
    badge_w, badge_h = 56, 28
    draw.rounded_rectangle(
        [(x, y), (x + badge_w, y + badge_h)],
        radius=6,
        fill=ACCENT,
    )
    draw.text(
        (x + badge_w / 2, y + badge_h / 2),
        f"§{panel.panel_id}",
        font=num_font,
        fill=BG,
        anchor="mm",
    )

    # Logo dot if this panel uses the brand symbol — secondary visual cue.
    if panel.use_logo:
        dot_x = x + badge_w + 10
        draw.ellipse(
            [(dot_x, y + 6), (dot_x + 16, y + 22)],
            fill=ACCENT,
            outline=ACCENT,
        )

    title_y = y + badge_h + 8
    draw.text((x, title_y), panel.title, font=title_font, fill=TEXT)

    if panel.caption:
        cap_y = title_y + 30
        for line in wrap_lines(draw, panel.caption, cap_font, w)[:3]:
            draw.text((x, cap_y), line, font=cap_font, fill=MUTED)
            cap_y += 24


def main() -> int:
    args = parse_args()
    if args.cols * args.rows < len(cb.PANELS):
        raise SystemExit(
            f"Layout too small: {args.cols}x{args.rows}={args.cols*args.rows}, "
            f"need {len(cb.PANELS)}."
        )

    panel_w = args.panel_width
    panel_h = int(panel_w * 2 / 3)
    cell_w = panel_w + args.margin
    cell_h = panel_h + args.caption_h + args.margin
    page_pad = args.margin * 2
    page_w = page_pad * 2 + cell_w * args.cols - args.margin
    page_h = page_pad * 2 + cell_h * args.rows - args.margin + 100

    canvas = Image.new("RGB", (page_w, page_h), BG)
    draw = ImageDraw.Draw(canvas)

    title_font = find_font(40, bold=True)
    sub_font = find_font(20)
    draw.text((page_pad, page_pad), "[Your Brand] — Cast a Shadow", font=title_font, fill=ACCENT)
    draw.text(
        (page_pad, page_pad + 50),
        "28 panels · GPT-Image-2 storyboard · "
        "[Your tagline goes here]",
        font=sub_font,
        fill=MUTED,
    )

    grid_top = page_pad + 100

    missing: list[str] = []
    for panel in cb.PANELS:
        col = (panel.idx - 1) % args.cols
        row = (panel.idx - 1) // args.cols
        x = page_pad + col * cell_w
        y = grid_top + row * cell_h

        draw.rectangle([(x, y), (x + panel_w, y + panel_h)], fill=PANEL_BG)
        if panel.path.exists():
            try:
                img = Image.open(panel.path).convert("RGB")
                img = img.resize((panel_w, panel_h), Image.LANCZOS)
                canvas.paste(img, (x, y))
            except Exception as exc:  # noqa: BLE001
                missing.append(f"{panel.path.name}: {exc}")
                draw.text(
                    (x + panel_w / 2, y + panel_h / 2),
                    f"err §{panel.panel_id}",
                    font=sub_font,
                    fill=MUTED,
                    anchor="mm",
                )
        else:
            missing.append(panel.path.name)
            draw.text(
                (x + panel_w / 2, y + panel_h / 2),
                f"missing §{panel.panel_id}",
                font=sub_font,
                fill=MUTED,
                anchor="mm",
            )

        cap_y = y + panel_h + 8
        draw_caption(canvas, panel, x, cap_y, panel_w)

    canvas.save(OUT_PATH, format="PNG", optimize=True)
    print(f"OK {OUT_PATH.relative_to(ROOT)} ({OUT_PATH.stat().st_size / 1_000_000:.1f} MB)")
    if missing:
        print(f"WARN {len(missing)} panel(s) missing or unreadable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
