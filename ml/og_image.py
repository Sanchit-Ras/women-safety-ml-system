"""The picture a link to the site shows when it is shared (LinkedIn, WhatsApp, X): 1200 x 630 pixels, today's district
map coloured as on the site's home page, beside the site's question. Run after ``ml.export_app``; the picture is
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
COLORS = {"fell": "#2563eb", "steady": "#94a3b8", "rose": "#fdba74", "rose_sharply": "#ea580c"}   # as frontend/src/lib/bands.js
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
    """The same words as the site's legend (frontend/src/lib/bands.js)."""
    out = []
    for i, b in enumerate(bands):
        if i == 0:
            label = f"Down more than {abs(round(b['below'] * 100))}%"
        elif b["below"] is None:
            label = f"Up {round(bands[i - 1]['below'] * 100)}% or more"
        else:
            sign = lambda x: f"{'+' if x > 0 else '−' if x < 0 else ''}{abs(round(x * 100))}%"   # noqa: E731
            label = f"{sign(bands[i - 1]['below'])} to {sign(b['below'])}"
        out.append((COLORS[b["band"]], label))
    return out


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
    band = {int(c): (d["summary"]["trend"] or {}).get("band") for c, d in app["districts"].items()}
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
        color = COLORS.get(band.get(code), NO_COMPARISON)
        for part in getattr(g, "geoms", [g]):
            d.polygon([to_px(*p) for p in part.exterior.coords], fill=color, outline="#ffffff", width=1)

    tr = app["meta"]["trend"]
    x0, width = 640, 520
    y = 120
    for line in wrap(d, "How has your district changed?", font("bold", 50), width):
        d.text((x0 * SCALE, y * SCALE), line, font=font("bold", 50), fill=INK)
        y += 62
    y += 14
    lines = wrap(d, f"Crimes against women reported in every district of India, {app['meta']['data_years'][0]}–"
                    f"{app['meta']['data_years'][1]}, rebuilt onto today's map so the years can be compared.", font("regular", 26), width)
    for line in lines:
        d.text((x0 * SCALE, y * SCALE), line, font=font("regular", 26), fill=MUTED)
        y += 36
    y += 28
    d.text((x0 * SCALE, y * SCALE), f"Change in reported cases, {tr['from_year']} to {tr['to_year']}", font=font("bold", 20), fill=INK)
    y += 38
    for color, label in legend(tr["bands"]):
        d.rounded_rectangle([x0 * SCALE, (y + 4) * SCALE, (x0 + 20) * SCALE, (y + 24) * SCALE], radius=4 * SCALE, fill=color)
        d.text(((x0 + 32) * SCALE, y * SCALE), label, font=font("regular", 21), fill=MUTED)
        y += 34
    d.text((x0 * SCALE, 584 * SCALE), "women-safety-ml-system-jwlh.vercel.app", font=font("regular", 20), fill="#2563eb")
    return img.resize(SIZE, Image.LANCZOS)


def main() -> None:
    app = json.loads(export_app.APP_DATA.read_text(encoding="utf-8"))
    geo = json.loads(export_app.MAP_OUT.read_text(encoding="utf-8"))
    draw(app, geo).save(OUT, optimize=True)
    print(f"{OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
