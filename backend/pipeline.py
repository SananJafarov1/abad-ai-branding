import os
import io
import json
import re
import base64
from pathlib import Path
from typing import Generator
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_TEXT_MODEL  = "gemini-3-flash-preview"
_IMAGE_MODEL = "nano-banana-pro-preview"
_JSON_SYSTEM_SUFFIX = (
    "Always respond in Azerbaijani. "
    "Respond ONLY with a valid JSON object, no markdown, no extra text."
)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                    if line.startswith("GEMINI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        _client = genai.Client(api_key=api_key)
    return _client


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"(\{[\s\S]*\})",
        r"(\[[\s\S]*\])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract valid JSON from response: {text[:200]}")


def _call_gemini(system: str, user: str) -> str:
    client = _get_client()
    response = client.models.generate_content(
        model=_TEXT_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=8000,
        ),
    )
    return response.text


def _generate_image(prompt: str, image_path: str | None = None) -> tuple[str, str]:
    client = _get_client()
    if image_path and Path(image_path).exists():
        img_bytes = Path(image_path).read_bytes()
        img_b64   = base64.b64encode(img_bytes).decode("utf-8")
        contents  = [
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ]
    else:
        contents = prompt
    response = client.models.generate_content(
        model=_IMAGE_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data is not None:
            return (
                base64.b64encode(part.inline_data.data).decode("utf-8"),
                part.inline_data.mime_type,
            )
    raise ValueError("No image generated")


def _ean13_checksum(digits12: str) -> int:
    total = sum(
        int(d) * (3 if i % 2 else 1)
        for i, d in enumerate(digits12)
    )
    return (10 - total % 10) % 10


def _draw_ean13(draw, barcode: str, cx: int, y: int, bar_w: int = 3, bar_h: int = 55):
    """Draw an EAN-13 barcode centred at cx, top at y."""
    L = {
        '0':'0001101','1':'0011001','2':'0010011','3':'0111101',
        '4':'0100011','5':'0110001','6':'0101111','7':'0111011',
        '8':'0110111','9':'0001011',
    }
    G = {
        '0':'0100111','1':'0110011','2':'0011011','3':'0100001',
        '4':'0011101','5':'0111001','6':'0000101','7':'0010001',
        '8':'0001001','9':'0010111',
    }
    R = {
        '0':'1110010','1':'1100110','2':'1101100','3':'1000010',
        '4':'1011100','5':'1001110','6':'1010000','7':'1000100',
        '8':'1001000','9':'1110100',
    }
    PARITY = {
        '0':'LLLLLL','1':'LLGLGG','2':'LLGGLG','3':'LLGGGL',
        '4':'LGLLGG','5':'LGGLLG','6':'LGGGLL','7':'LGLGLG',
        '8':'LGLGGL','9':'LGGLGL',
    }

    first        = barcode[0]
    left_digits  = barcode[1:7]
    right_digits = barcode[7:]
    parity       = PARITY[first]

    bits = '101'
    for i, d in enumerate(left_digits):
        bits += L[d] if parity[i] == 'L' else G[d]
    bits += '01010'
    for d in right_digits:
        bits += R[d]
    bits += '101'

    total_w = len(bits) * bar_w
    x = cx - total_w // 2

    for i, bit in enumerate(bits):
        if bit == '1':
            bx = x + i * bar_w
            draw.rectangle([bx, y, bx + bar_w - 1, y + bar_h], fill=(0, 0, 0, 255))


def _hex_to_rgba(hex_str: str, alpha: int = 255) -> tuple:
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (alpha,)


def _is_light(rgba: tuple) -> bool:
    """Returns True if the color is perceptually light (not readable as text on white)."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance > 160


def _darken(rgba: tuple, factor: float = 0.55) -> tuple:
    """Return a darker version of the color for use as text on white backgrounds."""
    return (int(rgba[0] * factor), int(rgba[1] * factor), int(rgba[2] * factor), 255)


def _on_color(bg_rgba: tuple) -> tuple:
    """Return white or dark gray depending on background luminance for readable text."""
    return (255, 255, 255, 255) if _is_light(bg_rgba) is False else (30, 30, 30, 255)


def _draw_wrapped(draw, text: str, font, x, y, max_w, color, line_h=20) -> int:
    words = text.split()
    line = ""
    for word in words:
        test = line + word + " "
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) > max_w and line:
            draw.text((x, y), line.strip(), font=font, fill=color)
            y += line_h
            line = word + " "
        else:
            line = test
    if line.strip():
        draw.text((x, y), line.strip(), font=font, fill=color)
        y += line_h
    return y


_FONT_CACHE: list | None = None

def _get_fonts():
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE
    from PIL import ImageFont
    _W = r"C:\Windows\Fonts"
    def _f(bold: bool, size: int):
        # Prefer Segoe UI (full Azerbaijani/Unicode support), fall back to Arial, then Calibri
        candidates = (["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"] if bold
                      else ["segoeui.ttf",  "arial.ttf",   "calibri.ttf"])
        for name in candidates:
            try:    return ImageFont.truetype(f"{_W}\\{name}", size)
            except: continue
        return ImageFont.load_default()
    _FONT_CACHE = [_f(True, 22), _f(True, 34), _f(True, 18),
                   _f(False, 15), _f(False, 13), _f(False, 11)]
    return _FONT_CACHE


def _safe_str(v) -> str:
    """Safely convert any AI-returned value (str, dict, list, etc.) to a plain string."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        # Try common text keys first
        for key in ("name", "text", "value", "description", "label"):
            if key in v and isinstance(v[key], str):
                return v[key].strip()
        # Fallback: first string value found
        for val in v.values():
            if isinstance(val, str):
                return val.strip()
        return str(v)
    if isinstance(v, list):
        return ", ".join(_safe_str(x) for x in v if x)
    return str(v) if v is not None else ""


_BEVERAGE_KEYWORDS = {
    "beverage", "drink", "juice", "lemonade", "limonad", "içki", "su", "water",
    "çay", "tea", "coffee", "qəhvə", "kompot", "şərbət", "mors", "nektar",
    "soda", "cola", "ayran", "kefir", "süd", "milk", "şirə", "meyvə suyu",
    "smoothie", "cocktail", "kokteyl", "mineral", "qazlı",
}

def _is_beverage(stage1: dict) -> bool:
    """Detect if product is a beverage based on category and name."""
    cat = (stage1.get("category") or "").lower()
    name = (stage1.get("name") or "").lower()
    text = f"{cat} {name}"
    print(f"  [_is_beverage] cat='{cat}' name='{name}'", flush=True)
    # Check multi-word keywords first
    _MULTI_WORD = {"meyvə suyu"}
    for mw in _MULTI_WORD:
        if mw in text:
            print(f"  [_is_beverage] MATCH multi-word: '{mw}'", flush=True)
            return True
    # Check single-word keywords with word boundaries
    words = set(text.split())
    match = words & _BEVERAGE_KEYWORDS
    if match:
        print(f"  [_is_beverage] MATCH words: {match}", flush=True)
    return bool(match)


def _create_back_label_pillow(bg_b64: str, stage1: dict, stage3: dict, stage4: dict) -> tuple[str, str]:
    """Overlay structured label text on AI-generated background using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    bg_data = base64.b64decode(bg_b64)
    bg_img  = Image.open(io.BytesIO(bg_data)).convert("RGBA")

    W, H = 630, 980
    bg_img = bg_img.resize((W, H), Image.LANCZOS)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    draw.rounded_rectangle([16, 16, W-16, H-16], radius=18, fill=(255, 255, 255, 238))

    colors  = stage3.get("colors", [])
    primary = _hex_to_rgba(colors[0]["hex"] if colors else "#1a6b3c")

    # If primary is too light to read on white, darken it for text use
    primary_text = _darken(primary) if _is_light(primary) else primary
    # Text color on top of primary-filled areas (table header etc.)
    on_primary   = _on_color(primary)

    f_brand, f_title, f_head, f_body, f_small, f_tiny = _get_fonts()

    PAD  = 44
    DARK = (30, 30, 30, 255)
    GRAY = (95, 95, 95, 255)
    y    = 38

    # ── Real data from label_text_generation ──────────────────────────────
    product_name = stage4.get("product_name") or stage1.get("name", "Məhsul")
    n            = stage4.get("nutrition") or {}
    kcal         = n.get("energy_kcal") or 0
    kj           = round(kcal * 4.184)
    protein      = n.get("protein_g",  0) or 0
    fat          = n.get("fat_g",      0) or 0
    carbs        = n.get("carbs_g",    0) or 0
    sugar        = n.get("sugar_g",    0) or 0
    salt         = n.get("salt_g",     0) or 0
    storage      = stage4.get("storage",   "") or "Sərin və quru yerdə saxlayın."
    shelf_life   = stage4.get("shelf_life", "") or "—"
    prod_date    = stage4.get("production_date_placeholder", "GG.AA.İİİİ")
    quantity     = stage4.get("quantity",  "") or "—"
    origin       = stage4.get("origin",   "Azərbaycan")
    manufacturer = stage4.get("manufacturer", "ABAD — Ailə Biznesinə Asan Dəstək Fərdi Təsərrüfatçılıq, Azərbaycan")
    allergens    = stage4.get("allergens", []) or []
    s4_ings      = stage4.get("ingredients", [])
    s1_ings      = stage1.get("ingredients", [])
    ings         = s4_ings if s4_ings else s1_ings
    ing_text     = ", ".join(
        str(i) if isinstance(i, str)
        else f"{i.get('name','')}" + (f" ({i.get('percentage','')}%)" if i.get('percentage') else "")
        for i in ings
    ) or "—"

    # ── Brand ──────────────────────────────────────────────────────────────
    draw.text((PAD, y), "ABAD", font=f_brand, fill=primary_text)
    y += 32

    # ── Product name ────────────────────────────────────────────────────────
    draw.text((PAD, y), product_name, font=f_title, fill=(18, 18, 18, 255))
    y += 46

    # ── Divider ─────────────────────────────────────────────────────────────
    draw.line([(PAD, y), (W-PAD, y)], fill=primary_text, width=2)
    y += 14

    # ── Ingredients ─────────────────────────────────────────────────────────
    draw.text((PAD, y), "TƏRKİB:", font=f_head, fill=primary_text)
    y += 24
    y = _draw_wrapped(draw, ing_text, f_body, PAD, y, W - PAD*2, DARK, 20)
    y += 10

    # ── Divider ─────────────────────────────────────────────────────────────
    draw.line([(PAD, y), (W-PAD, y)], fill=(180, 180, 180, 200), width=1)
    y += 12

    # ── Nutrition table ──────────────────────────────────────────────────────
    unit = "ml" if _is_beverage(stage1) else "q"
    draw.text((PAD, y), f"QİDALILIQ DƏYƏRİ (100 {unit}-a görə):", font=f_head, fill=primary_text)
    y += 26

    TW = W - PAD * 2
    draw.rectangle([PAD, y, PAD+TW, y+22], fill=primary)
    draw.text((PAD+8, y+4), "Göstərici", font=f_small, fill=on_primary)
    draw.text((PAD+TW-105, y+4), "Miqdar", font=f_small, fill=on_primary)
    y += 22

    rows = [
        ("Enerji",         f"{kj} kC / {kcal} kkal"),
        ("Zülal",          f"{protein} q"),
        ("Yağ",            f"{fat} q"),
        ("Karbohidratlar", f"{carbs} q"),
        ("Şəkər",          f"{sugar} q"),
        ("Duz",            f"{salt} q"),
    ]
    alt = False
    for lbl, val in rows:
        fill = (235, 248, 238, 210) if alt else (255, 255, 255, 210)
        draw.rectangle([PAD, y, PAD+TW, y+20], fill=fill)
        draw.text((PAD+8, y+3), lbl, font=f_small, fill=DARK)
        draw.text((PAD+TW-105, y+3), val, font=f_small, fill=DARK)
        y += 20
        alt = not alt
    y += 12

    # ── Allergens ───────────────────────────────────────────────────────────
    if allergens:
        draw.rounded_rectangle([PAD, y, W-PAD, y+32], radius=6, fill=(255, 243, 205, 230))
        alg_text = "⚠ Allergen: " + ", ".join(allergens)
        draw.text((PAD+10, y+7), alg_text, font=f_small, fill=(160, 60, 0, 255))
        y += 40

    # ── Divider ─────────────────────────────────────────────────────────────
    draw.line([(PAD, y), (W-PAD, y)], fill=(180, 180, 180, 200), width=1)
    y += 12

    # ── Storage & info ───────────────────────────────────────────────────────
    draw.text((PAD, y), "SAXLANMA:", font=f_head, fill=primary_text)
    y += 22
    y = _draw_wrapped(draw, storage, f_body, PAD, y, W - PAD*2, DARK, 19)
    y += 8

    info_lines = [
        f"İstehsal tarixi: {prod_date}",
        f"Yararlılıq müddəti: {shelf_life}",
        f"Xalis çəki / Həcm: {quantity}",
        f"Mənşəyi: {origin}",
    ]
    for line in info_lines:
        draw.text((PAD, y), line, font=f_small, fill=GRAY)
        y += 18
    y += 8

    # ── Divider ─────────────────────────────────────────────────────────────
    draw.line([(PAD, y), (W-PAD, y)], fill=(180, 180, 180, 200), width=1)
    y += 12

    # ── Manufacturer ────────────────────────────────────────────────────────
    draw.text((PAD, y), "İSTEHSALÇI:", font=f_head, fill=primary_text)
    y += 22
    y = _draw_wrapped(draw, manufacturer, f_small, PAD, y, W - PAD*2, GRAY, 16)
    y += 10

    # ── EAN-13 Barcode (476 = Azerbaijan) ───────────────────────────────────
    bc_digits12 = "476000000000"
    checksum    = _ean13_checksum(bc_digits12)
    barcode     = bc_digits12 + str(checksum)

    bc_y   = H - 105
    bc_cx  = W // 2
    _draw_ean13(draw, barcode, bc_cx, bc_y, bar_w=3, bar_h=52)

    # Barcode number below bars
    bc_label = f"{barcode[:1]}  {barcode[1:7]}  {barcode[7:]}"
    bbox = draw.textbbox((0, 0), bc_label, font=f_tiny)
    lw   = bbox[2] - bbox[0]
    draw.text((bc_cx - lw//2, bc_y + 56), bc_label, font=f_tiny, fill=(40, 40, 40, 255))

    # ── Composite & save ─────────────────────────────────────────────────────
    result = Image.alpha_composite(bg_img, overlay)
    buf    = io.BytesIO()
    result.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"


def _overlay_logo_on_front(front_b64: str, logo_path: Path) -> tuple[str, str]:
    """Paste the real ABAD logo onto the top-center of the AI-generated front label."""
    from PIL import Image
    import numpy as np

    # Decode AI-generated front label
    front_data = base64.b64decode(front_b64)
    front_img = Image.open(io.BytesIO(front_data)).convert("RGBA")
    fw, fh = front_img.size

    # Load and resize logo to fit ~40% of label width
    logo = Image.open(str(logo_path)).convert("RGBA")
    logo_w = int(fw * 0.4)
    logo_h = int(logo.height * logo_w / logo.width)
    logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

    # Remove white/near-white background from logo → make transparent
    data = np.array(logo)
    white_mask = (data[:, :, 0] > 230) & (data[:, :, 1] > 230) & (data[:, :, 2] > 230)
    data[white_mask, 3] = 0
    logo = Image.fromarray(data)

    # Paste at top center with some padding
    x = (fw - logo_w) // 2
    y = int(fh * 0.02)  # 2% from top
    front_img.paste(logo, (x, y), logo)  # use logo alpha as mask

    buf = io.BytesIO()
    front_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"


def _generate_label_images(stage1: dict, stage3: dict, stage4: dict) -> dict:
    """Generate front (AI) and back (AI bg + Pillow text) label images."""
    product_name = _safe_str(stage4.get("product_name") or stage1.get("name", "Məhsul")) or "Məhsul"
    category     = _safe_str(stage1.get("category", "food product")) or "food product"
    ings_preview = ", ".join(
        _safe_str(i) for i in stage1.get("ingredients", [])[:5] if i
    ) or product_name
    colors       = stage3.get("colors", [])
    primary      = colors[0].get("hex", "#2d9b5a") if colors else "#2d9b5a"
    secondary    = colors[1].get("hex", "#c9952a") if len(colors) > 1 else "#c9952a"
    visual_style = _safe_str(stage3.get("visual_style", "")) or "natural, artisanal"
    selling_pts  = ", ".join(
        _safe_str(s) for s in stage1.get("selling_points", [])[:3] if s
    ) or ""

    # ── Front label (AI generates artwork, then Pillow overlays real logo) ───
    front_prompt = (
        f"Create a flat premium product LABEL sticker design for '{product_name}'. "
        f"The label is a standalone flat design on plain white background. "
        f"NOT a bottle, NOT a jar, NOT any 3D mockup — just the flat label sticker itself. "
        f"\n\nSTRICT RULES:\n"
        f"1. Do NOT include any logo, brand name, icon, or emblem. The real logo will be overlaid later. "
        f"   Fill the ENTIRE label with artwork — no blank/empty areas.\n"
        f"2. The ONLY text on the entire label must be: '{product_name}' — written in large, elegant typography in the center.\n"
        f"3. ABSOLUTELY NO English text anywhere. No ingredient names, no annotations, no descriptions, no captions. "
        f"   NOTHING except '{product_name}'. No 'Fresh', 'Natural', 'Pure', 'Organic'.\n"
        f"4. NO arrows, NO labels, NO text callouts, NO annotations of any kind.\n"
        f"5. Decorative watercolor illustrations of {product_name} as purely visual decoration only.\n"
        f"\nVISUAL STYLE: Watercolor illustrations of {product_name}. "
        f"Color palette: primary {primary}, accent {secondary}. "
        f"Portrait orientation 9:14 ratio, artisanal high-end look, cream or white label background."
    )

    # ── Back label background (AI, no text at all) ──────────────────────────
    back_bg_prompt = (
        f"Soft decorative background for a '{product_name}' ({category}) food product label. "
        f"Use subtle watercolor illustrations of {ings_preview} as repeating pattern elements. "
        f"Colors: {primary} and {secondary}. Style: {visual_style}. "
        "Very subtle, low-opacity background art — must not overpower text placed on top. "
        "Absolutely NO text, NO writing, NO letters, NO numbers anywhere. "
        "Pure background artwork only. Portrait 2:3 ratio."
    )

    # ── Debug: print resolved prompt variables to server console ─────────────
    def _p(label, val):
        try:
            print(f"  {label}: {val}")
        except UnicodeEncodeError:
            print(f"  {label}: {str(val).encode('ascii', errors='replace').decode('ascii')}")
    print("\n=== LABEL IMAGE GENERATION DEBUG ===")
    _p("product_name", product_name)
    _p("category    ", category)
    _p("ings_preview", ings_preview)
    _p("visual_style", visual_style)
    _p("selling_pts ", selling_pts)
    _p("primary     ", f"{primary}  secondary: {secondary}")
    _p("FRONT PROMPT", front_prompt)
    print("=====================================\n")

    _logo_path = Path(__file__).resolve().parent.parent / "frontend" / "label_logo" / "labe_logo-removebg-preview.png"

    # Generate both images in parallel (no logo passed to AI — we overlay it ourselves)
    results = [None, None]
    def _gen_front(): results[0] = _generate_image(front_prompt)
    def _gen_back():  results[1] = _generate_image(back_bg_prompt)
    t1 = __import__('threading').Thread(target=_gen_front, daemon=True)
    t2 = __import__('threading').Thread(target=_gen_back,  daemon=True)
    t1.start(); t2.start(); t1.join(); t2.join()
    front_b64, front_mime = results[0]
    back_bg_b64, _        = results[1]

    # Resize front label and overlay ABAD logo with oval frame using Pillow
    from PIL import Image as _PILImage, ImageDraw as _PILDraw
    import numpy as _np

    _front_data = base64.b64decode(front_b64)
    _front_img = _PILImage.open(io.BytesIO(_front_data)).convert("RGBA")

    # Auto-crop white margins so the artwork fills the frame
    _arr = _np.array(_front_img)
    _non_white = (_arr[:, :, 0] < 245) | (_arr[:, :, 1] < 245) | (_arr[:, :, 2] < 245)
    _rows = _np.any(_non_white, axis=1)
    _cols = _np.any(_non_white, axis=0)
    if _rows.any() and _cols.any():
        _rmin, _rmax = _np.where(_rows)[0][[0, -1]]
        _cmin, _cmax = _np.where(_cols)[0][[0, -1]]
        _front_img = _front_img.crop((_cmin, _rmin, _cmax + 1, _rmax + 1))

    W, H = _front_img.size

    # Load logo and resize to ~20% of label width
    _logo = _PILImage.open(str(_logo_path)).convert("RGBA")
    _logo_w = int(W * 0.20)
    _logo_h = int(_logo.height * _logo_w / _logo.width)
    _logo = _logo.resize((_logo_w, _logo_h), _PILImage.LANCZOS)

    # Logo already has transparent background — no removal needed

    # Oval frame position and size
    _pad_x, _pad_y = int(_logo_w * 0.35), int(_logo_h * 0.30)
    _oval_w = _logo_w + _pad_x * 2
    _oval_h = _logo_h + _pad_y * 2
    _oval_cx = W // 2
    _oval_cy = int(H * 0.14)

    # Sample average background color from the oval area for a seamless blend
    _sample_points = []
    for _sx in range(_oval_cx - _oval_w // 4, _oval_cx + _oval_w // 4, 5):
        for _sy in range(_oval_cy - _oval_h // 4, _oval_cy + _oval_h // 4, 5):
            if 0 <= _sx < W and 0 <= _sy < H:
                _sample_points.append(_front_img.getpixel((_sx, _sy)))
    if _sample_points:
        _bg = tuple(int(sum(c) / len(_sample_points)) for c in zip(*_sample_points))
    else:
        _bg = _front_img.getpixel((_oval_cx, _oval_cy))

    # Draw filled oval
    _draw = _PILDraw.Draw(_front_img)
    _oval_box = [
        _oval_cx - _oval_w // 2, _oval_cy - _oval_h // 2,
        _oval_cx + _oval_w // 2, _oval_cy + _oval_h // 2,
    ]
    _draw.ellipse(_oval_box, fill=(_bg[0], _bg[1], _bg[2], 255))

    # Paste logo centered inside the oval
    _lx = _oval_cx - _logo_w // 2
    _ly = _oval_cy - _logo_h // 2
    _front_img.paste(_logo, (_lx, _ly), _logo)

    _buf = io.BytesIO()
    _front_img.convert("RGB").save(_buf, format="PNG")
    front_b64 = base64.b64encode(_buf.getvalue()).decode("utf-8")
    front_mime = "image/png"

    # Pillow overlays all real label text on back background
    back_b64, back_mime = _create_back_label_pillow(back_bg_b64, stage1, stage3, stage4)

    return {
        "front": {"image_base64": front_b64, "mime_type": front_mime},
        "back":  {"image_base64": back_b64,  "mime_type": back_mime},
    }


def _run_stage(
    stage_name: str, system: str, user: str, fallback: dict | None = None
) -> tuple[dict, str]:
    raw = None
    try:
        raw    = _call_gemini(system, user)
        result = extract_json(raw)
        chunk  = json.dumps(
            {"stage": stage_name, "status": "complete", "data": result},
            ensure_ascii=False,
        )
        return result, chunk
    except Exception as e:
        error_data = {"error": str(e), "raw": raw or ""}
        chunk = json.dumps(
            {"stage": stage_name, "status": "error", "data": error_data},
            ensure_ascii=False,
        )
        if fallback is not None:
            return fallback, chunk
        raise RuntimeError(chunk) from e


def run_pipeline_phase1(product_input: str) -> Generator[str, None, None]:
    """Stages 1-3: analysis, market research, design. Pauses for user colour selection."""

    # ── Stage 1: Product Analysis ────────────────────────────────────────────
    stage1, chunk = _run_stage(
        "product_analysis",
        f"You are an expert product analyst for ABAD Azerbaijan. {_JSON_SYSTEM_SUFFIX}",
        (
            f'Analyze this product: "{product_input}". '
            "Provide: 1) Full product name, 2) Category, 3) Target market (Azerbaijan), "
            "4) Key selling points, 5) Typical ingredients/components. "
            "Format as JSON with keys: name, category, target_market, "
            "selling_points (array), ingredients (array)."
        ),
        fallback={
            "name": product_input, "category": "Naməlum",
            "target_market": "Azərbaycan", "selling_points": [], "ingredients": [],
        },
    )
    yield chunk

    # ── Stage 2: Market Research ─────────────────────────────────────────────
    stage2, chunk = _run_stage(
        "market_research",
        f"You are a market research expert for Azerbaijan food/craft market. {_JSON_SYSTEM_SUFFIX}",
        (
            f"Based on this product: {json.dumps(stage1, ensure_ascii=False)}. "
            "Provide market research: 1) Market size/opportunity in Azerbaijan, "
            "2) Main competitors and their positioning, 3) Price range recommendations, "
            "4) Consumer preferences, 5) Seasonal trends. "
            "Format as JSON with keys: market_opportunity, "
            "competitors (array of {name, positioning}), price_range, "
            "consumer_preferences, seasonal_trends."
        ),
        fallback={},
    )
    yield chunk

    # ── Stage 3: Design Recommendations ──────────────────────────────────────
    stage3, chunk = _run_stage(
        "design_recommendations",
        f"You are a brand design expert specializing in Azerbaijani products. {_JSON_SYSTEM_SUFFIX}",
        (
            f"Based on product {json.dumps(stage1, ensure_ascii=False)} "
            f"and market research {json.dumps(stage2, ensure_ascii=False)}. "
            "Provide: 1) Color palette (5 colors with hex codes and names in Azerbaijani), "
            "2) Typography style recommendations, 3) Visual style/mood, "
            "4) Packaging shape/material suggestions, 5) Logo concept ideas. "
            "Format as JSON with keys: colors (array of {hex, name, usage}), "
            "typography (plain string), visual_style (plain string), packaging (plain string), "
            "logo_concepts (array of plain strings, not objects)."
        ),
        fallback={},
    )
    yield chunk


def run_pipeline_phase2(
    stage1: dict, stage2: dict, stage3: dict, user_colors: list
) -> Generator[str, None, None]:
    """Stages 4-7 using user-confirmed colours."""

    # Override AI colours with user selection
    stage3 = {**stage3, "colors": user_colors}
    product_input = stage1.get("name", "Məhsul")

    # ── Stage 4: Label Text Generation ───────────────────────────────────────
    stage4, chunk = _run_stage(
        "label_text_generation",
        (
            "You are a professional Azerbaijani copywriter and proofreader specializing in food product labels. "
            "STRICT LANGUAGE RULES: "
            "- Use only Azerbaijani language (Latin alphabet). "
            "- Zero spelling mistakes. Zero grammatical errors. "
            "- Use correct Azerbaijani characters (ə, ö, ü, ğ, ş, ç, ı). "
            "- Text must sound natural and professional, like real product packaging. "
            "- Avoid repetition and unnecessary words. Keep sentences clear and concise. "
            "- Do NOT translate word-by-word — adapt naturally to Azerbaijani. "
            "- Before finalizing, double-check and correct ALL Azerbaijani grammar and spelling. "
            f"{_JSON_SYSTEM_SUFFIX}"
        ),
        (
            f"Product type: {stage1.get('name', product_input)}. "
            f"Features: {', '.join(stage1.get('selling_points', []))}. "
            f"Ingredients: {', '.join(str(i) if isinstance(i, str) else i.get('name','') for i in stage1.get('ingredients', []))}. "
            f"Target audience: {stage1.get('target_market', 'Azərbaycan əhalisi')}. "
            "Generate a complete product label with PERFECT Azerbaijani grammar and spelling. "
            "Output as JSON with these keys: "
            "product_name (short, clear, catchy), "
            "tagline (1 natural slogan sentence), "
            "description (1-2 natural professional sentences, max 50 words), "
            "ingredients (array of {name, percentage} — realistic values, percentages must sum near 100), "
            "nutrition (object: energy_kcal, protein_g, fat_g, carbs_g, sugar_g, salt_g — ALL numeric, use 0 if negligible, never null), "
            "storage (clear storage instruction), "
            "production_date_placeholder (e.g. 'GG.AA.İİİİ'), "
            "shelf_life (e.g. '12 ay'), "
            "manufacturer ('ABAD — Ailə Biznesinə Asan Dəstək Fərdi Təsərrüfatçılıq' with address template), "
            "allergens (array of strings, empty array if none), "
            "quantity (net weight/volume string, e.g. '300 q'), "
            "origin ('Azərbaycan')."
        ),
        fallback={},
    )
    yield chunk

    # ── Stage 5: Label Images (AI front + Pillow back with real data) ─────────
    try:
        images = _generate_label_images(stage1, stage3, stage4)
        chunk  = json.dumps(
            {"stage": "label_image", "status": "complete", "data": images},
            ensure_ascii=False,
        )
    except Exception as e:
        chunk = json.dumps(
            {"stage": "label_image", "status": "error", "data": {"error": str(e)}},
            ensure_ascii=False,
        )
    yield chunk

    # ── Stage 6: Compliance Check ─────────────────────────────────────────────
    _, chunk = _run_stage(
        "compliance_check",
        f"You are a regulatory compliance expert for Azerbaijan food safety law and AQTA. {_JSON_SYSTEM_SUFFIX}",
        (
            f"Check this label for compliance: {json.dumps(stage4, ensure_ascii=False)}. "
            "Check against: 1) Article 17.2.1 of Azerbaijan Republic Law on Food Safety "
            "(product name, nutritional value, composition, quantity, production date, "
            "shelf life, storage conditions, allergens, origin, manufacturer name/address), "
            "2) ГОСТ standards, 3) AZS standards, 4) ISO standards, 5) AQTA requirements. "
            'Format as JSON with keys: overall_status ("UYĞUN"/"UYĞUN DEYİL"/"QISMƏN UYĞUN"), '
            "law_compliance (object with each requirement as key, value: {status: bool, notes: str}), "
            "gost_compliance, azs_compliance, iso_compliance, aqta_compliance, "
            'issues (array of {severity: "kritik"/"xəbərdarlıq"/"məlumat", description, recommendation}), '
            "recommendations (array of strings)."
        ),
        fallback={},
    )
    yield chunk

    # ── Stage 7: Language Check ───────────────────────────────────────────────
    _, chunk = _run_stage(
        "language_check",
        f"You are an Azerbaijani language expert and orthographer. {_JSON_SYSTEM_SUFFIX}",
        (
            f"Review this label text: {json.dumps(stage4, ensure_ascii=False)}. "
            "Check: 1) Spelling errors, 2) Grammar issues, "
            "3) Terminology consistency, 4) Formal/professional tone, 5) Clarity. "
            'Format as JSON with keys: overall_quality ("Əla"/"Yaxşı"/"Qənaətbəxş"/"Zəif"), '
            "spelling_score (0-100), grammar_score (0-100), "
            "errors (array of {original, corrected, explanation}), "
            "improvements (array of strings), final_verdict (string)."
        ),
        fallback={},
    )
    yield chunk
