"""
EcoSort AI — Waste Item Classifier
====================================
NLP-based classifier that matches user-supplied item names against the
waste catalog.  Falls back to fuzzy string matching when an exact keyword
hit is not found.  Includes a lifecycle calculation engine for landfill
space and decomposition metrics.

Also provides:
  - ``classify_from_image()``     — single-item visual classification via
      filename heuristics with hint-text override.
  - ``area_audit_from_image()``   — full garbage-area audit using
      Gemini Vision / Groq Vision with deterministic heuristic fallback
      (delegates to ``core.vision_engine``).

SDG Alignment: SDG 12 (Responsible Consumption & Production)
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Load .env at import time so any os.environ reads in this module and in
# core.vision_engine (imported lazily) see GEMINI_API_KEY when it is set
# in the project .env file rather than the shell environment.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on env vars set externally

# ---------------------------------------------------------------------------
# Optional dependency: difflib (always available in stdlib)
# ---------------------------------------------------------------------------
from difflib import SequenceMatcher

# Catalogue path (relative to project root)
_CATALOG_PATH = Path(__file__).parent.parent / "data" / "waste_catalog.json"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Holds the full classification output for a single waste item query."""

    matched: bool
    item_id: str = ""
    item_name: str = ""
    category_key: str = ""
    category_display: str = ""
    bin_color: str = ""
    bin_hex: str = "#6b7280"
    primary_material: str = ""
    decomposition_years: float = 0.0
    recycling_steps: list[str] = field(default_factory=list)
    handling_tips: str = ""
    hazard_level: str = "none"
    regulatory_note: str = ""
    carbon_offset_kg_per_kg: float = 0.0
    weight_kg_per_unit: float = 0.0
    landfill_space_cm3_per_unit: float = 0.0
    confidence: float = 0.0
    match_method: str = ""  # "exact", "alias", "fuzzy", "unmatched", "visual"
    lifecycle: dict = field(default_factory=dict)
    # Image analysis metadata (populated only when classify_from_image is used)
    visual_tags: list[str] = field(default_factory=list)
    visual_query: str = ""


@dataclass
class LifecycleStats:
    """Derived lifecycle metrics for a given quantity of waste."""

    item_name: str
    quantity_units: int
    total_weight_kg: float
    decomposition_years: float
    landfill_volume_litres: float
    carbon_offset_kg: float
    carbon_offset_trees_equivalent: float  # 1 tree ≈ 21 kg CO₂/yr
    landfill_years_saved: float            # assuming avg 10 m³ landfill cell


# ---------------------------------------------------------------------------
# Visual heuristics table
# ---------------------------------------------------------------------------
# Maps visual keyword tokens (from filename or simulated object-detection tags)
# to catalog search queries.  Entries are checked in order; first match wins.
# Format: (set_of_trigger_tokens, catalog_query_string, display_label)
#
# In a production system this table would be populated from a real CV model's
# label vocabulary (e.g. Google Vision, YOLOv8 class names).  The mock here
# covers the full waste-catalog item set so that every catalog item is
# reachable via a plausible filename or tag.
_VISUAL_HEURISTICS: list[tuple[frozenset[str], str, str]] = [
    # Organic / Wet
    (frozenset({"banana", "peel", "skin", "rind"}),         "banana peel",        "🍌 Banana Peel"),
    (frozenset({"food", "scrap", "leftover", "kitchen",
                "vegetable", "fruit", "organic"}),           "food scraps",        "🥦 Food Scraps"),
    (frozenset({"garden", "leaf", "leaves", "grass",
                "twig", "yard", "plant", "trim"}),           "garden waste",       "🌿 Garden Waste"),
    (frozenset({"egg", "shell", "eggshell"}),                "egg shells",         "🥚 Egg Shells"),
    (frozenset({"tea", "bag", "teabag"}),                    "tea bags",           "🍵 Tea Bags"),
    # Recyclable / Dry
    # Note: aluminium/soda/can must precede plastic bottle so that "soda can"
    # filenames resolve to Aluminium Can rather than Plastic Bottle.
    (frozenset({"aluminium", "aluminum", "can", "tin",
                "soda", "beer", "metal"}),                   "aluminium can",      "🥤 Aluminium Can"),
    (frozenset({"plastic", "bottle", "pet", "hdpe",
                "container", "water"}),                      "plastic bottle",     "🍶 Plastic Bottle"),
    (frozenset({"cardboard", "carton", "box", "corrugated",
                "shipping", "package"}),                     "cardboard box",      "📦 Cardboard Box"),
    (frozenset({"glass", "jar", "wine",
                "bottle"}),                                  "glass bottle",       "🍾 Glass Bottle"),
    (frozenset({"paper", "newspaper", "magazine",
                "journal", "print"}),                        "newspaper",          "📰 Newspaper"),
    (frozenset({"steel", "iron", "scrap", "ferrous"}),       "steel scrap",        "🔩 Steel Scrap"),
    # E-Waste
    (frozenset({"phone", "mobile", "smartphone", "iphone",
                "android", "cellphone"}),                    "smartphone",         "📱 Smartphone"),
    (frozenset({"laptop", "notebook", "macbook",
                "computer"}),                                "laptop",             "💻 Laptop"),
    (frozenset({"monitor", "crt", "screen", "display",
                "cathode"}),                                 "crt monitor",        "🖥️ CRT Monitor"),
    (frozenset({"cartridge", "ink", "toner", "printer"}),   "printer cartridge",  "🖨️ Printer Cartridge"),
    (frozenset({"battery", "lithium", "liion",
                "powerbank", "rechargeable"}),               "lithium battery",    "🔋 Lithium Battery"),
    # Toxic / Hazardous
    # Motor oil must precede Paint so "motor_oil" filenames don't hit Paint.
    (frozenset({"motor", "engine", "lubricant",
                "crankcase"}),                               "motor oil",          "🛢️ Motor Oil"),
    (frozenset({"paint", "latex", "spray",
                "aerosol", "varnish"}),                      "paint",              "🎨 Paint"),
    (frozenset({"fluorescent", "tube", "cfl", "bulb",
                "neon", "mercury", "lamp"}),                 "fluorescent tube",   "💡 Fluorescent Tube"),
    (frozenset({"pesticide", "insecticide", "herbicide",
                "chemical"}),                                "pesticide container","⚗️ Pesticide Container"),
    (frozenset({"needle", "syringe", "sharp", "lancet",
                "medical", "insulin"}),                      "medical sharps",     "💉 Medical Sharps"),
]

# Tokens that are too generic to be informative (ignored during heuristic scan)
_STOP_TOKENS: frozenset[str] = frozenset({
    "img", "image", "photo", "pic", "picture", "jpg", "jpeg", "png",
    "scan", "capture", "cam", "camera", "upload", "file", "test",
    "new", "old", "used", "item", "waste", "trash", "garbage",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
})


def _extract_visual_tags(filename: str) -> list[str]:
    """
    Tokenise an image filename into lowercase words, stripping the extension,
    underscores, hyphens, and digits-only tokens.

    Example: "old_plastic_bottle_001.jpg" → ["old", "plastic", "bottle"]
    """
    stem = Path(filename).stem
    raw_tokens = re.split(r"[^a-z]+", stem.lower())
    return [t for t in raw_tokens if len(t) >= 3 and t not in _STOP_TOKENS]


def infer_query_from_tags(tags: list[str]) -> tuple[str, str]:
    """
    Match a list of visual tags against the heuristics table.

    Returns
    -------
    (catalog_query, display_label)
        catalog_query is the string passed to WasteClassifier.classify().
        display_label is a human-readable emoji label for the UI.
        Both are empty strings if no heuristic matched.
    """
    tag_set = frozenset(tags)
    for trigger_tokens, query, label in _VISUAL_HEURISTICS:
        if trigger_tokens & tag_set:          # any intersection → match
            return query, label
    return "", ""


# ---------------------------------------------------------------------------
# Catalog loader (cached at module level)
# ---------------------------------------------------------------------------

_catalog_cache: Optional[dict] = None


def _load_catalog() -> dict:
    """Load and cache the waste catalog JSON."""
    global _catalog_cache
    if _catalog_cache is None:
        if not _CATALOG_PATH.exists():
            raise FileNotFoundError(
                f"Waste catalog not found at {_CATALOG_PATH}. "
                "Ensure data/waste_catalog.json is present."
            )
        with open(_CATALOG_PATH, "r", encoding="utf-8") as fh:
            _catalog_cache = json.load(fh)
    return _catalog_cache


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation/extra spaces."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    """Return SequenceMatcher ratio between two normalised strings."""
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Core Classifier
# ---------------------------------------------------------------------------

class WasteClassifier:
    """
    Classifies a waste item by:
    1. Exact name match (case-insensitive).
    2. Alias / keyword match.
    3. Fuzzy string similarity fallback (threshold ≥ 0.55).

    Usage::

        clf = WasteClassifier()
        result = clf.classify("old phone")
        print(result.bin_color, result.recycling_steps)
    """

    FUZZY_THRESHOLD = 0.55  # minimum similarity to accept a fuzzy match

    def __init__(self) -> None:
        catalog = _load_catalog()
        self._categories: dict = catalog.get("categories", {})
        # Pre-build a flat lookup table for fast matching
        self._lookup: list[dict] = self._build_lookup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, raw_query: str, quantity: int = 1) -> ClassificationResult:
        """
        Classify a waste item query.

        Parameters
        ----------
        raw_query : str
            Free-text item name (e.g. "banana peel", "old laptop").
        quantity : int
            Number of units — used for lifecycle calculation.

        Returns
        -------
        ClassificationResult
            Full classification with lifecycle metrics.
        """
        if not raw_query or not raw_query.strip():
            return ClassificationResult(matched=False, match_method="empty_query")

        query_norm = _normalise(raw_query)

        # 1. Exact name match
        result = self._exact_match(query_norm)

        # 2. Alias / keyword match
        if result is None:
            result = self._alias_match(query_norm)

        # 3. Fuzzy fallback
        if result is None:
            result = self._fuzzy_match(query_norm)

        if result is None:
            return ClassificationResult(
                matched=False,
                match_method="unmatched",
                confidence=0.0,
            )

        # Attach lifecycle calculations
        result.lifecycle = self._compute_lifecycle(result, quantity).__dict__
        return result

    # ------------------------------------------------------------------
    # Image classification
    #
    # Priority chain for a call that carries actual image bytes:
    #   1. Gemini Vision (gemini-1.5-flash) — if GEMINI_API_KEY is set
    #   2. hint_text / manual_override text — if provided
    #   3. Filename heuristic tag lookup
    #   4. Safe fallback → Plastic Bottle (confidence 0.35)
    #
    # Calling conventions (both still supported):
    #   Keyword:    classify_from_image(filename="bottle.jpg", quantity=1,
    #                                   manual_override="...")
    #   Positional: classify_from_image(image_bytes, hint_text="old phone")
    # ------------------------------------------------------------------

    # Gemini prompt for single-item classification
    _GEMINI_ITEM_PROMPT = (
        "Analyze this waste/garbage image. "
        "Identify what items are visible, categorize them "
        "(Wet/Organic, Dry/Recyclable, E-Waste, or Hazardous), "
        "suggest the exact disposal bin color (Green, Blue, Red, or Black), "
        "and give a 2-line handling recommendation. "
        "Reply in this exact format with no extra text:\n"
        "ITEM: <primary waste item name>\n"
        "CATEGORY: <Wet/Organic | Dry/Recyclable | E-Waste | Hazardous>\n"
        "BIN: <Green | Blue | Red | Black>\n"
        "TIPS: <2-line handling recommendation>"
    )

    # Ordered list of vision-capable model names to try.
    # The first model that responds without a 404/unsupported error is used.
    _GEMINI_MODELS = [
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro-vision",
    ]

    @staticmethod
    def _normalise_image_bytes(image_bytes: bytes) -> bytes:
        """
        Ensure the image is JPEG-encoded RGB so every Gemini model accepts it.

        Some camera captures or PNGs with alpha channels cause model errors.
        Converting to RGB and re-encoding as JPEG is a safe common denominator.
        Falls back to returning the original bytes if PIL is not installed.
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
            return image_bytes  # PIL unavailable or parse failed — use as-is

    @staticmethod
    def _call_gemini_for_item(image_bytes: bytes) -> str:
        """
        Send *image_bytes* to the best available Gemini vision model and
        return the raw text response.

        Tries each entry in ``_GEMINI_MODELS`` in order.  A model is skipped
        when the API returns a 404 / "not found" / "not supported" response;
        any other error (invalid key, quota, network) is re-raised immediately
        so ``app.py`` can surface it via ``st.error()``.

        The image is converted to JPEG RGB before sending so all models
        accept it regardless of original format or colour mode.

        Raises
        ------
        EnvironmentError
            When ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` is not set.
        RuntimeError
            When all models in the fallback list are exhausted (all 404).
        Exception
            Any non-404 SDK exception (invalid key, quota, network).
        """
        import os
        import warnings
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or set it as an environment variable."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai as genai

        genai.configure(api_key=api_key)

        # Normalise to JPEG RGB — safe for all vision models
        safe_bytes = WasteClassifier._normalise_image_bytes(image_bytes)
        from core.vision_engine import _detect_mime
        mime = _detect_mime(safe_bytes)
        image_part = {"mime_type": mime, "data": safe_bytes}

        last_err: Optional[Exception] = None
        for model_name in WasteClassifier._GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    [WasteClassifier._GEMINI_ITEM_PROMPT, image_part],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=300,
                    ),
                )
                return response.text.strip()
            except Exception as exc:
                msg = str(exc).lower()
                # 404 / not-found / not-supported → try next model silently
                if any(k in msg for k in ("404", "not found", "not supported",
                                          "not exist", "deprecated")):
                    last_err = exc
                    continue
                # Any other error (auth, quota, network) → raise immediately
                raise

        # All models exhausted
        raise RuntimeError(
            f"No working Gemini vision model found. "
            f"Tried: {WasteClassifier._GEMINI_MODELS}. "
            f"Last error: {last_err}"
        )

    @staticmethod
    def _parse_gemini_item_response(text: str) -> tuple[str, str, str, str]:
        """
        Parse the structured Gemini response into (item, category, bin, tips).
        Returns empty strings for any field that cannot be parsed.
        """
        import re as _re
        def _field(key: str) -> str:
            m = _re.search(rf"^{key}:\s*(.+)", text, _re.MULTILINE | _re.IGNORECASE)
            return m.group(1).strip() if m else ""

        item     = _field("ITEM")
        category = _field("CATEGORY")
        bin_col  = _field("BIN")
        # TIPS may span two lines — grab everything after "TIPS:"
        m_tips = _re.search(r"^TIPS:\s*(.+(?:\n.+)?)", text, _re.MULTILINE | _re.IGNORECASE)
        tips = m_tips.group(1).strip() if m_tips else ""
        return item, category, bin_col, tips

    def classify_from_image(
        self,
        image_data_or_filename=None,
        hint_text: Optional[str] = None,
        *,
        # Keyword-only aliases kept for backward-compat with existing app.py calls
        filename: Optional[str] = None,
        quantity: int = 1,
        manual_override: Optional[str] = None,
    ) -> ClassificationResult:
        """
        Classify a single waste item from an uploaded or camera-captured image.

        **Priority chain:**

        1. **Gemini Vision** — if real image bytes are present and
           ``GEMINI_API_KEY`` is set, the image is sent to
           ``gemini-1.5-flash`` with a structured item-identification prompt.
           The response is parsed and matched against the catalog.
        2. **hint_text / manual_override** — free-text override, classified
           via the standard NLP pipeline.
        3. **Filename heuristic** — tokenise the filename and run the
           visual-heuristics tag table.
        4. **Safe fallback** — returns Plastic Bottle at confidence 0.35 so
           the caller always receives a well-formed ``ClassificationResult``.

        Parameters
        ----------
        image_data_or_filename : bytes | bytearray | BinaryIO | str | None
            Raw image bytes **or** an image filename string.
        hint_text : str, optional
            Free-text item name hint (maps to ``manual_override``).
        filename : str, optional
            Image filename (keyword-only).
        quantity : int
            Unit count for lifecycle calculations.
        manual_override : str, optional
            Alias for ``hint_text`` (keyword-only, for app.py compatibility).

        Returns
        -------
        ClassificationResult
            Fully populated result including lifecycle metrics.
            ``matched`` is always ``True`` (fallback guarantees it).
        """
        import os as _os

        # ── Resolve calling convention ─────────────────────────────────
        resolved_filename: str = "unknown_image.jpg"
        resolved_hint: Optional[str] = hint_text or manual_override
        image_bytes: Optional[bytes] = None

        if image_data_or_filename is not None:
            if isinstance(image_data_or_filename, str):
                resolved_filename = image_data_or_filename
            else:
                # bytes / bytearray / file-like
                resolved_filename = getattr(image_data_or_filename, "name", "camera_image.jpg")
                if isinstance(image_data_or_filename, (bytes, bytearray)):
                    image_bytes = bytes(image_data_or_filename)
                elif hasattr(image_data_or_filename, "read"):
                    image_bytes = image_data_or_filename.read()
        elif filename:
            resolved_filename = filename

        # ── Step 1: Gemini Vision (real image bytes → AI detection) ────
        # Raises on missing key or API failure — propagated to app.py for
        # display via st.error() so the exact error is always visible.
        if image_bytes and len(image_bytes) > 100:
            raw_text = self._call_gemini_for_item(image_bytes)   # may raise
            item_name, category, bin_col, tips = self._parse_gemini_item_response(raw_text)
            if item_name:
                # Try to match the Gemini-detected item against catalog
                catalog_result = self.classify(item_name, quantity=quantity)
                if catalog_result.matched:
                    # Catalog match — use catalog data, enrich with Gemini tips
                    catalog_result.visual_tags = _extract_visual_tags(resolved_filename)
                    catalog_result.visual_query = item_name
                    catalog_result.match_method = "gemini"
                    catalog_result.confidence = 0.92
                    if tips:
                        catalog_result.handling_tips = tips
                    return catalog_result
                else:
                    # Gemini identified something not in catalog — build inline result
                    bin_hex_map = {
                        "green": "#22c55e", "blue": "#3b82f6",
                        "red": "#ef4444",   "black": "#1f2937",
                    }
                    bin_col_norm = bin_col.lower() if bin_col else "blue"
                    bin_hex = bin_hex_map.get(bin_col_norm, "#3b82f6")
                    gemini_result = ClassificationResult(
                        matched=True,
                        item_id="GEMINI-DETECTED",
                        item_name=item_name,
                        category_display=category or "Unclassified",
                        bin_color=bin_col.capitalize() if bin_col else "Blue",
                        bin_hex=bin_hex,
                        handling_tips=tips or "Follow local waste disposal guidelines.",
                        confidence=0.85,
                        match_method="gemini",
                        visual_tags=_extract_visual_tags(resolved_filename),
                        visual_query=item_name,
                    )
                    gemini_result.lifecycle = self._compute_lifecycle(
                        gemini_result, quantity
                    ).__dict__
                    return gemini_result

        # ── Step 2: hint / override text ───────────────────────────────
        if resolved_hint and resolved_hint.strip():
            result = self.classify(resolved_hint.strip(), quantity=quantity)
            result.visual_tags = _extract_visual_tags(resolved_filename)
            result.visual_query = resolved_hint.strip()
            return result

        # ── Step 3: filename heuristic ─────────────────────────────────
        tags = _extract_visual_tags(resolved_filename)
        inferred_query, display_label = infer_query_from_tags(tags)
        if inferred_query:
            result = self.classify(inferred_query, quantity=quantity)
            result.visual_tags = tags
            result.visual_query = display_label or inferred_query
            if result.matched:
                result.match_method = "visual"
            return result

        # ── Step 4: safe fallback ──────────────────────────────────────
        fallback = self.classify("plastic bottle", quantity=quantity)
        fallback.visual_tags = tags
        fallback.visual_query = "Generic Dry Recyclable (fallback)"
        fallback.confidence = 0.35
        fallback.match_method = "visual_fallback"
        return fallback

    def area_audit_from_image(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
    ):
        """
        Perform a full garbage-area audit on an image of a waste dump,
        street garbage, or mixed trash scene.

        Delegates to ``core.vision_engine.analyse_garbage_area`` which tries:
          1. Google Gemini Vision (if ``GEMINI_API_KEY`` env var is set)
          2. Groq Vision        (if ``GROQ_API_KEY`` env var is set)
          3. Heuristic fallback (always available — never raises)

        Parameters
        ----------
        image_bytes : bytes
            Raw image bytes from ``st.file_uploader`` or ``st.camera_input``.
        filename : str
            Original filename — used by heuristic for context.

        Returns
        -------
        core.vision_engine.GarbageAuditReport
            Complete report with detected items, segregation percentages,
            action plan, and hazard level.
        """
        from core.vision_engine import analyse_garbage_area
        return analyse_garbage_area(image_bytes, filename=filename)

    def list_all_items(self) -> list[dict]:
        """Return a flat list of all catalog items with category info."""
        items = []
        for cat_key, cat_data in self._categories.items():
            for item in cat_data.get("items", []):
                items.append({
                    "category": cat_data.get("display_name", cat_key),
                    "bin_color": cat_data.get("bin_color", ""),
                    **item,
                })
        return items

    def get_category_summary(self) -> list[dict]:
        """Return per-category summary for dashboard use."""
        summary = []
        for cat_key, cat_data in self._categories.items():
            summary.append({
                "key": cat_key,
                "display_name": cat_data.get("display_name", cat_key),
                "bin_color": cat_data.get("bin_color", ""),
                "bin_hex": cat_data.get("bin_hex", "#6b7280"),
                "item_count": len(cat_data.get("items", [])),
                "handling_summary": cat_data.get("handling_summary", ""),
            })
        return summary

    # ------------------------------------------------------------------
    # Private matching helpers
    # ------------------------------------------------------------------

    def _build_lookup(self) -> list[dict]:
        """Flatten the catalog into a searchable list."""
        lookup = []
        for cat_key, cat_data in self._categories.items():
            for item in cat_data.get("items", []):
                lookup.append({
                    "cat_key": cat_key,
                    "cat_display": cat_data.get("display_name", cat_key),
                    "bin_color": cat_data.get("bin_color", ""),
                    "bin_hex": cat_data.get("bin_hex", "#6b7280"),
                    "item": item,
                    "name_norm": _normalise(item.get("name", "")),
                    "aliases_norm": [_normalise(a) for a in item.get("aliases", [])],
                })
        return lookup

    def _build_result(self, entry: dict, confidence: float, method: str) -> ClassificationResult:
        """Construct a ClassificationResult from a lookup entry."""
        item = entry["item"]
        return ClassificationResult(
            matched=True,
            item_id=item.get("id", ""),
            item_name=item.get("name", ""),
            category_key=entry["cat_key"],
            category_display=entry["cat_display"],
            bin_color=entry["bin_color"],
            bin_hex=entry["bin_hex"],
            primary_material=item.get("primary_material", ""),
            decomposition_years=item.get("decomposition_years", 0.0),
            recycling_steps=item.get("recycling_steps", []),
            handling_tips=item.get("handling_tips", ""),
            hazard_level=item.get("hazard_level", "none"),
            regulatory_note=item.get("regulatory_note", ""),
            carbon_offset_kg_per_kg=item.get("carbon_offset_kg_per_kg", 0.0),
            weight_kg_per_unit=item.get("weight_kg_per_unit", 0.0),
            landfill_space_cm3_per_unit=item.get("landfill_space_cm3_per_unit", 0.0),
            confidence=confidence,
            match_method=method,
        )

    def _exact_match(self, query_norm: str) -> Optional[ClassificationResult]:
        for entry in self._lookup:
            if entry["name_norm"] == query_norm:
                return self._build_result(entry, 1.0, "exact")
        return None

    def _alias_match(self, query_norm: str) -> Optional[ClassificationResult]:
        """Check if query appears in any alias list (substring or full match)."""
        best: Optional[tuple[float, dict]] = None
        for entry in self._lookup:
            for alias in entry["aliases_norm"]:
                # Full alias match
                if alias == query_norm:
                    return self._build_result(entry, 0.97, "alias")
                # Substring containment (query inside alias or alias inside query)
                if query_norm in alias or alias in query_norm:
                    score = len(min(alias, query_norm, key=len)) / len(
                        max(alias, query_norm, key=len)
                    )
                    if best is None or score > best[0]:
                        best = (score, entry)
        if best and best[0] >= 0.6:
            return self._build_result(best[1], best[0] * 0.95, "alias")
        return None

    def _fuzzy_match(self, query_norm: str) -> Optional[ClassificationResult]:
        """Return best fuzzy match above FUZZY_THRESHOLD."""
        best_score = 0.0
        best_entry: Optional[dict] = None

        for entry in self._lookup:
            candidates = [entry["name_norm"]] + entry["aliases_norm"]
            for candidate in candidates:
                score = _similarity(query_norm, candidate)
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry is not None and best_score >= self.FUZZY_THRESHOLD:
            return self._build_result(best_entry, best_score * 0.85, "fuzzy")
        return None

    # ------------------------------------------------------------------
    # Lifecycle engine
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_lifecycle(result: ClassificationResult, quantity: int) -> LifecycleStats:
        """
        Derive environmental impact metrics for `quantity` units of the item.

        Formulas:
        - Landfill volume (litres) = landfill_space_cm3 * qty / 1000
        - Carbon offset (kg) = carbon_offset_kg_per_kg * weight_kg * qty
        - Tree equivalent = carbon_offset / 21  (avg kg CO₂ absorbed per tree/yr)
        - Landfill years saved = landfill_volume / 10_000  (10 m³ cell = 10_000 L)
        """
        qty = max(1, int(quantity))
        total_weight = result.weight_kg_per_unit * qty
        landfill_vol_litres = (result.landfill_space_cm3_per_unit * qty) / 1000.0
        co2_offset = result.carbon_offset_kg_per_kg * total_weight
        tree_equiv = co2_offset / 21.0
        landfill_years_saved = landfill_vol_litres / 10_000.0

        return LifecycleStats(
            item_name=result.item_name,
            quantity_units=qty,
            total_weight_kg=round(total_weight, 4),
            decomposition_years=result.decomposition_years,
            landfill_volume_litres=round(landfill_vol_litres, 4),
            carbon_offset_kg=round(co2_offset, 4),
            carbon_offset_trees_equivalent=round(tree_equiv, 4),
            landfill_years_saved=round(landfill_years_saved, 8),
        )
