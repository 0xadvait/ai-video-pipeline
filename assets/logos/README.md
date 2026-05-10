# Brand logos

Drop your logo PNG here as `Symbol_Cyan.png`. The scripts will pick it up
automatically — it's referenced as `LOGO_CYAN = LOGO_DIR / "Symbol_Cyan.png"`
in `scripts/generate_shadow_comic.py`.

## Specs that work well

- **Format:** PNG with transparency (RGBA)
- **Size:** at least 256×256, preferably 512×512 or larger
- **Color:** the brand-mark color you want to be the only saturated color
  in the final film. The default scripts use `#24bce3` (a cyan) — change
  the `BRAND_CYAN` constant in `generate_shadow_comic.py` if your brand
  is a different hex.
- **Geometry:** any clear single-mark symbol works (a letter, a glyph, a
  geometric mark). The film's narrative beats lean on the agent's shadow
  *being* this symbol, so picking a recognizable mark is what makes the
  ending land.

## What the scripts do with it

| Script | How the logo is used |
|---|---|
| `generate_shadow_comic.py` | Passed as `input_images=[bible, logo]` on every panel that has `use_logo=True`. GPT-Image-2 then draws the brand mark literally into the panel where the prompt asks for it. |
| `generate_shadow_transitions.py` | The video clips inherit the panel's logo geometry through their first/last frames. No extra logo input needed. |

## If you want a placeholder for now

Any 512×512 PNG of any 4-petal-style geometric mark will do for testing
the pipeline. Re-run `python3 scripts/generate_shadow_comic.py --only 13 16 19 20 26 27 28 --force` to regenerate just the panels that show the symbol.
