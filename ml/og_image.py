"""The picture a link to the site shows when it is shared (LinkedIn, WhatsApp, X): 1200 x 630 pixels, today's district
map coloured by the 22-year change, as on the site's home page, beside the site's question. Run after ``ml.export_app``; the picture is
committed and served with the site (frontend/public/og.png, named in frontend/index.html).

    python -m ml.og_image
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import shape

from ml import export_app

OUT = export_app.ROOT / "frontend" / "public" / "og.png"
SIZE = (1200, 630)
SCALE = 2                       # drawn at twice the size, then shrunk: smooth edges
MAP_BOX = (40, 25, 600, 605)    # left, top, right, bottom of the map, in final pixels
# As frontend/src/lib/bands.js: blue fell, grey changed little, oranges rose (one more orange for the 22-year view)
PALETTES = {4: ["#2563eb", "#94a3b8", "#fdba74", "#ea580c"], 5: ["#2563eb", "#94a3b8", "#fdba74", "#f97316", "#c2410c"]}
NO_COMPARISON = "#eef2f6"
INK, MUTED = "#0f172a", "#475569"
FONTS = {"bold": ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
         "regular": ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]}


def font(kind: str, size: int):
    for path in FONTS[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size * SCALE)
    return ImageFont.load_default(size * SCALE)


def mercator(lon: float, lat: float) -> tuple[float, float]:
    return math.radians(lon), math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def legend(bands: list[dict]) -> list[tuple[str, str]]:
    """(colour, words) per band, as the site's legend shows them."""
    return [(color, b["label"][0].upper() + b["label"][1:]) for color, b in zip(PALETTES[len(bands)], bands)]


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) <= width * SCALE:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def draw(app: dict, geo: dict) -> Image.Image:
    w = app["meta"]["trends"]["long"]
    colors = dict((b["label"], color) for b, (color, _label) in zip(w["bands"], legend(w["bands"])))
    band = {int(c): d["summary"]["long"]["band"] for c, d in app["districts"].items()          # coloured only over all the years
            if d["summary"]["long"] and d["summary"]["long"]["from_year"] == w["from_year"]}
    shapes = [(f["properties"]["code"], shape(f["geometry"])) for f in geo["features"]]
    shapes.sort(key=lambda s: -s[1].area)                  # large first, so a district inside another is drawn on top
    xs, ys = zip(*(mercator(x, y) for _c, g in shapes for x, y in (g.bounds[:2], g.bounds[2:])))
    left, top, right, bottom = MAP_BOX
    k = min((right - left) / (max(xs) - min(xs)), (bottom - top) / (max(ys) - min(ys))) * SCALE
    ox = (left * SCALE) + ((right - left) * SCALE - k * (max(xs) - min(xs))) / 2

    def to_px(lon, lat):
        x, y = mercator(lon, lat)
        return ox + k * (x - min(xs)), top * SCALE + k * (max(ys) - y)

    img = Image.new("RGB", (SIZE[0] * SCALE, SIZE[1] * SCALE), "#ffffff")
    d = ImageDraw.Draw(img)
    for code, g in shapes:
        color = colors.get(band.get(code), NO_COMPARISON)
        for part in getattr(g, "geoms", [g]):
            d.polygon([to_px(*p) for p in part.exterior.coords], fill=color, outline="#ffffff", width=1)

    span = w["to_year"] - w["from_year"] + 1
    x0, width = 640, 520
    y = 64
    for line in wrap(d, f"How has your district changed in {span} years?", font("bold", 46), width):
        d.text((x0 * SCALE, y * SCALE), line, font=font("bold", 46), fill=INK)
        y += 56
    y += 12
    lines = wrap(d, f"Crimes against women reported in every district of India, every year from {w['from_year']} to "
                    f"{w['to_year']}, rebuilt onto today's map.", font("regular", 24), width)
    for line in lines:
        d.text((x0 * SCALE, y * SCALE), line, font=font("regular", 24), fill=MUTED)
        y += 33
    y += 22
    d.text((x0 * SCALE, y * SCALE), f"Change in reported cases, {w['from_year']} to {w['to_year']}", font=font("bold", 20), fill=INK)
    y += 36
    for color, label in legend(w["bands"]):
        d.rounded_rectangle([x0 * SCALE, (y + 4) * SCALE, (x0 + 20) * SCALE, (y + 24) * SCALE], radius=4 * SCALE, fill=color)
        d.text(((x0 + 32) * SCALE, y * SCALE), label, font=font("regular", 21), fill=MUTED)
        y += 31
    d.text((x0 * SCALE, 584 * SCALE), "women-safety-ml-system-jwlh.vercel.app", font=font("regular", 20), fill="#2563eb")
    return img.resize(SIZE, Image.LANCZOS)


def main() -> None:
    app = json.loads(export_app.APP_DATA.read_text(encoding="utf-8"))
    geo = json.loads(export_app.MAP_OUT.read_text(encoding="utf-8"))
    draw(app, geo).save(OUT, optimize=True)
    print(f"{OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
