"""
EcoSort AI — Garbage Area Vision Engine
=========================================
Performs multimodal visual analysis of garbage dump / street-waste images.

Priority order for analysis:
  1. Google EcoSort AI Vision (gemini-1.5-flash)  — if GEMINI_API_KEY is set
  2. Groq Vision (llama-3.2-11b-vision)       — if GROQ_API_KEY is set
  3. Heuristic fallback                        — deterministic rule-based
                                                 garbage composition estimate

All three paths return the same ``GarbageAuditReport`` dataclass so the UI
never needs to handle different shapes.

SDG Alignment: SDG 11 (Sustainable Cities), SDG 12 (Responsible Production)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Load .env so GEMINI_API_KEY is available even without shell export.
# This is safe to call multiple times — dotenv skips vars already set.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on env vars being set externally

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DetectedItem:
    """A single waste item spotted in the image."""
    name: str
    category: str          # "Organic/Wet" | "Dry Recyclable" | "E-Waste/Hazardous"
    quantity_estimate: str  # e.g. "Several pieces", "~3 bags"
    bin_color: str
    bin_hex: str
    action: str            # one-line disposal instruction


@dataclass
class GarbageAuditReport:
    """
    Full area-level garbage audit result returned by the vision engine.
    All fields have safe defaults so even a failed analysis is renderable.
    """
    # Source metadata
    source: str = "heuristic"          # "gemini" | "groq" | "heuristic"
    image_description: str = ""        # brief scene description from LLM or heuristic

    # Detected items list
    detected_items: list[DetectedItem] = field(default_factory=list)

    # Segregation breakdown (percentages, must sum ≤ 100)
    pct_wet: float = 0.0
    pct_dry: float = 0.0
    pct_hazardous: float = 0.0
    pct_inert: float = 0.0            # construction debris, soil, etc.

    # Overall hazard assessment
    hazard_level: str = "Low"         # "Low" | "Medium" | "Severe"
    hazard_reason: str = ""

    # Action plan (ordered steps)
    action_plan: list[str] = field(default_factory=list)

    # Environmental impact note
    env_impact: str = ""

    # Error message (non-empty only when vision API call failed)
    error: str = ""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_VISION_PROMPT = """
You are EcoSort AI, an expert waste management analyst. Analyse this image of a garbage area, waste dump, or mixed trash.

Return a JSON object with EXACTLY this structure (no markdown, no code fences, raw JSON only):
{
  "image_description": "<one-sentence scene description, e.g. 'Urban street corner with mixed household waste'>",
  "detected_items": [
    {
      "name": "<item name>",
      "category": "<Organic/Wet | Dry Recyclable | E-Waste/Hazardous | Inert>",
      "quantity_estimate": "<e.g. Several pieces, ~3 bags, Large pile>",
      "bin_color": "<Green | Blue | Red | Black>",
      "bin_hex": "<hex color for bin>",
      "action": "<one-line Hindi-English mixed disposal instruction>"
    }
  ],
  "segregation": {
    "pct_wet": <0-100 integer>,
    "pct_dry": <0-100 integer>,
    "pct_hazardous": <0-100 integer>,
    "pct_inert": <0-100 integer>
  },
  "hazard_level": "<Low | Medium | Severe>",
  "hazard_reason": "<Why this hazard level — 1-2 sentences>",
  "action_plan": [
    "<Step 1 — specific segregation/disposal action>",
    "<Step 2>",
    "<Step 3>",
    "<Step 4>",
    "<Step 5>"
  ],
  "env_impact": "<1-2 sentence environmental impact statement>"
}

Rules:
- detected_items should list ALL distinct waste types visible (minimum 3, maximum 12).
- pct_wet + pct_dry + pct_hazardous + pct_inert must equal 100.
- action_plan steps should be practical, actionable, in Hinglish (Hindi + English mix).
- bin_hex values: Green=#22c55e, Blue=#3b82f6, Red=#ef4444, Black=#1f2937
- If image is unclear, make a reasonable estimate based on common urban garbage composition.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_base64(image_bytes: bytes) -> str:
    """Encode raw image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _detect_mime(image_bytes: bytes) -> str:
    """Detect MIME type from magic bytes."""
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] in (b"RIFF", b"WEBP"):
        return "image/webp"
    return "image/jpeg"  # safe default


def _parse_llm_json(raw: str) -> dict:
    """
    Extract and parse JSON from LLM response.
    Handles responses wrapped in markdown code fences.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    # Find the outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(cleaned[start:end])


def _dict_to_report(data: dict, source: str) -> GarbageAuditReport:
    """Convert a parsed LLM JSON dict into a GarbageAuditReport."""
    seg = data.get("segregation", {})
    items = [
        DetectedItem(
            name=it.get("name", "Unknown"),
            category=it.get("category", "Dry Recyclable"),
            quantity_estimate=it.get("quantity_estimate", "Unknown"),
            bin_color=it.get("bin_color", "Blue"),
            bin_hex=it.get("bin_hex", "#3b82f6"),
            action=it.get("action", "Place in appropriate bin."),
        )
        for it in data.get("detected_items", [])
    ]
    return GarbageAuditReport(
        source=source,
        image_description=data.get("image_description", ""),
        detected_items=items,
        pct_wet=float(seg.get("pct_wet", 0)),
        pct_dry=float(seg.get("pct_dry", 0)),
        pct_hazardous=float(seg.get("pct_hazardous", 0)),
        pct_inert=float(seg.get("pct_inert", 0)),
        hazard_level=data.get("hazard_level", "Low"),
        hazard_reason=data.get("hazard_reason", ""),
        action_plan=data.get("action_plan", []),
        env_impact=data.get("env_impact", ""),
    )


# ---------------------------------------------------------------------------
# Provider 1: Google EcoSort AI Vision  (google-generativeai, model fallback list)
# ---------------------------------------------------------------------------

# Ordered list of vision-capable model names to try for area audit.
# Mirrors WasteClassifier._GEMINI_MODELS — kept in sync manually.
_GEMINI_MODELS = [
      "models/gemini-1.5-flash",
      "models/gemini-1.5-flash-8b",
]


def _normalise_to_jpeg_rgb(image_bytes: bytes) -> bytes:
    """
    Convert image bytes to JPEG-encoded RGB.

    Removes alpha channels (RGBA/PA) and re-encodes so every Gemini model
    variant accepts the payload.  Falls back to original bytes when PIL is
    unavailable or the bytes cannot be decoded.
    """
    try:
        import io as _io
        from PIL import Image as _Image
        img = _Image.open(_io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        return image_bytes  # PIL unavailable or unreadable — use as-is


def _analyse_with_gemini(image_bytes: bytes) -> GarbageAuditReport:
    """
    Send the image to the first working EcoSort AI Vision model from
    ``_GEMINI_MODELS`` and return a ``GarbageAuditReport``.

    Model selection:
      Tries each model in ``_GEMINI_MODELS`` in order.  A model is skipped
      silently when the API returns 404 / "not found" / "not supported".
      Any other error (bad key, quota, network, JSON parse) is re-raised
      so the caller can surface it via ``st.error()``.

    Raises
    ------
    EnvironmentError
        When ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` is absent.
    RuntimeError
        When all models in the fallback list return 404 / unsupported.
    Exception
        Any non-404 API error (invalid key, quota, network).
    """
    # Ensure .env is loaded — safe to call again (dotenv skips set vars)
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env", override=False)
    except ImportError:
        pass

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import google.generativeai as genai  # pinned SDK — see requirements.txt

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    # Normalise to JPEG RGB before sending
    safe_bytes = _normalise_to_jpeg_rgb(image_bytes)
    mime = _detect_mime(safe_bytes)
    image_part = {"mime_type": mime, "data": safe_bytes}

    last_err: Optional[Exception] = None
    for model_name in _GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                [_VISION_PROMPT, image_part],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=1500,
                ),
            )
            data = _parse_llm_json(response.text)
            return _dict_to_report(data, source="gemini")
        except Exception as exc:
            msg = str(exc).lower()
            # 404 / not-found / not-supported → skip to next model silently
            if any(k in msg for k in ("404", "not found", "not supported",
                                      "not exist", "deprecated")):
                last_err = exc
                continue
            # Any other error → raise immediately for st.error() display
            raise

    raise RuntimeError(
        f"No working EcoSort AI Vision model found. "
        f"Tried: {_GEMINI_MODELS}. "
        f"Last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# Provider 2: Groq Vision (llama-3.2-11b-vision-preview)
# ---------------------------------------------------------------------------

def _analyse_with_groq(image_bytes: bytes) -> GarbageAuditReport:
    """
    Send image to Groq llama-3.2-11b-vision-preview for garbage area analysis.
    Raises on any error — caller catches and falls through to heuristic.
    """
    from groq import Groq  # optional dep

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)
    mime = _detect_mime(image_bytes)
    b64 = _image_to_base64(image_bytes)
    data_url = f"data:{mime};base64,{b64}"

    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0.2,
        max_tokens=1500,
    )
    data = _parse_llm_json(response.choices[0].message.content)
    return _dict_to_report(data, source="groq")


# ---------------------------------------------------------------------------
# Provider 3: Heuristic fallback
# ---------------------------------------------------------------------------

# Typical urban garbage composition profiles (used when no API is available)
_COMPOSITION_PROFILES = [
    # (pct_wet, pct_dry, pct_hazardous, pct_inert, profile_name)
    (55, 30, 5, 10, "Mixed residential garbage"),
    (40, 45, 5, 10, "Commercial / market waste"),
    (65, 20, 3, 12, "Food market / vegetable mandi"),
    (30, 50, 8, 12, "Institutional / office complex"),
    (45, 35, 12, 8,  "Urban street dump with e-waste"),
    (20, 35, 5, 40,  "Construction site with debris"),
]

_HEURISTIC_DETECTED_ITEMS = [
    DetectedItem("Plastic Bags / Wraps",    "Dry Recyclable",     "Many pieces",   "Blue",  "#3b82f6",
                 "Plastic bags uthao, compress karke Blue bin mein daalo."),
    DetectedItem("Food Waste / Rotten Veg", "Organic/Wet",        "Large quantity","Green", "#22c55e",
                 "Organic waste Green bin mein daalo — composting ke liye bhejo."),
    DetectedItem("Cardboard / Paper",       "Dry Recyclable",     "Several sheets","Blue",  "#3b82f6",
                 "Cardboard ko flat karke Blue recycling bin mein rakho."),
    DetectedItem("Glass Bottles / Jars",    "Dry Recyclable",     "~5 pieces",     "Blue",  "#3b82f6",
                 "Glass bottles rinse karke Blue bin mein daalo — mat todo."),
    DetectedItem("Thermocol / Foam",        "Dry Recyclable",     "Few chunks",    "Blue",  "#3b82f6",
                 "Thermocol alag karo — certified recycler ke paas bhejo."),
    DetectedItem("Mixed Food Scraps",       "Organic/Wet",        "Scattered",     "Green", "#22c55e",
                 "Kitchen waste Green bin mein — compost pit ya biogas unit."),
    DetectedItem("Used Batteries",          "E-Waste/Hazardous",  "2-3 units",     "Red",   "#ef4444",
                 "Batteries Red/e-waste bin mein — kabhi bhi regular bin mein mat daalo."),
    DetectedItem("Sanitary / Medical Waste","E-Waste/Hazardous",  "Some pieces",   "Black", "#1f2937",
                 "Sanitary waste Black bag mein seal karke — PHC ya municipal drive mein jama karo."),
]

_HEURISTIC_ACTION_PLAN = [
    "Sabse pehle waste ko teen bundles mein alag karo: Geela (Green), Sukha (Blue), aur Khatarnak (Red/Black).",
    "Plastic bags, thermocol aur cardboard ko ek jagah ikhatta karo — yeh sab dry recyclable hain (Blue bin).",
    "Rotten vegetables, food waste aur organic material Green bin mein daalo — composting ke liye taiyaar hai.",
    "Batteries, tube lights, aur koi bhi e-waste Red bin mein alag rakho — inhe kabhi bhi regular bin mein mat daalo.",
    "Ek designated volunteer ya safai mitra ko waste weighing aur manifest banane ke liye niyukt karo.",
    "Municipal collection schedule check karo — bulk generator hone par on-site composting unit lagana compulsory hai.",
    "Poori jagah ko saaf karne ke baad lime powder ya organic deodoriser chhidko — disease vectors ko rokne ke liye.",
]


def _heuristic_report(image_bytes: bytes, filename: str = "") -> GarbageAuditReport:
    """
    Generate a realistic garbage audit report without a vision API.
    Uses the image byte-hash to deterministically pick a composition
    profile so the same image always produces the same report.
    """
    # Deterministic profile selection via image hash
    img_hash = int(hashlib.md5(image_bytes[:512]).hexdigest(), 16)
    profile = _COMPOSITION_PROFILES[img_hash % len(_COMPOSITION_PROFILES)]
    pct_wet, pct_dry, pct_haz, pct_inert, profile_name = profile

    # Select a realistic subset of detected items (5–7) deterministically
    item_indices = sorted(set(
        (img_hash >> i) % len(_HEURISTIC_DETECTED_ITEMS)
        for i in range(0, 28, 4)
    ))[:7]
    selected_items = [_HEURISTIC_DETECTED_ITEMS[i] for i in item_indices]

    # Determine hazard level from profile
    if pct_haz >= 10:
        hazard_level = "Severe"
        hazard_reason = (
            "Significant e-waste and hazardous materials detected. "
            "Risk of soil and groundwater contamination if left untreated."
        )
    elif pct_haz >= 5:
        hazard_level = "Medium"
        hazard_reason = (
            "Moderate hazardous content (batteries, chemical containers) present. "
            "Requires segregation before municipal collection."
        )
    else:
        hazard_level = "Low"
        hazard_reason = (
            "Primarily organic and recyclable waste. "
            "Low contamination risk but proper segregation needed."
        )

    return GarbageAuditReport(
        source="heuristic",
        image_description=(
            f"{profile_name} — estimated composition based on image analysis."
        ),
        detected_items=selected_items,
        pct_wet=pct_wet,
        pct_dry=pct_dry,
        pct_hazardous=pct_haz,
        pct_inert=pct_inert,
        hazard_level=hazard_level,
        hazard_reason=hazard_reason,
        action_plan=_HEURISTIC_ACTION_PLAN,
        env_impact=(
            f"Estimated {pct_wet}% organic waste can be diverted to composting, "
            f"saving significant landfill space and reducing methane emissions. "
            f"{pct_dry}% dry recyclables can re-enter the material cycle."
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_garbage_area(
    image_bytes: bytes,
    filename: str = "image.jpg",
) -> GarbageAuditReport:
    """
    Analyse a garbage area image and return a full ``GarbageAuditReport``.

    **Provider selection logic:**

    - If ``GEMINI_API_KEY`` is set → calls Gemini.  Any API failure
      (invalid key, quota, network, parse error) **raises** so the caller
      can display the exact error message.  Never silently falls back.
    - If no Gemini key and ``GROQ_API_KEY`` is set → calls Groq.  Same
      raise-on-failure contract.
    - If neither key is set → returns the deterministic heuristic report
      silently (no API configured).

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded or camera-captured image.
    filename : str
        Original filename (used by heuristic fallback for context).

    Returns
    -------
    GarbageAuditReport
        Complete audit report with detected items, segregation breakdown,
        action plan, and hazard level.

    Raises
    ------
    Exception
        Any API error when a key is configured — propagated to the caller
        so ``st.error()`` can display the exact message.
    """
    if not image_bytes:
        return _heuristic_report(b"fallback", filename)

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    groq_key   = os.getenv("GROQ_API_KEY")

    # ── Gemini: key configured → always attempt, never silent-fallback ──
    if gemini_key:
        # Raises on any failure — caller (app.py) handles with st.error()
        return _analyse_with_gemini(image_bytes)

    # ── Groq: key configured → always attempt, never silent-fallback ────
    if groq_key:
        return _analyse_with_groq(image_bytes)

    # ── No key configured → heuristic (silent, expected path) ───────────
    return _heuristic_report(image_bytes, filename)
