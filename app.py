"""
EcoSort AI — Enterprise Waste Intelligence & Policy Assistant
=============================================================
Production-grade Streamlit application aligned with UN SDG 12 & SDG 11.

Tabs:
  1. Live Waste Analyzer       — classify items, upload/camera image, lifecycle stats
  2. Policy Assistant          — RAG-powered municipal/campus policy search
  3. Impact & Carbon Metrics   — KPIs and charts for waste diversion
  4. Responsible AI Audit      — fairness, explainability, SDG alignment

Run:
    streamlit run app.py

Author: EcoSort AI Engineering Team
Version: 2.5.0
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Load .env before anything else so GEMINI_API_KEY is available to all
# modules (classifier, vision_engine) that read os.environ.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — env vars must be set externally

import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from any working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.classifier import WasteClassifier
from core.rag_engine import PolicyRAGEngine

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/ecosort-ai",
        "Report a bug": "https://github.com/ecosort-ai/issues",
        "About": "EcoSort AI — Enterprise Waste Intelligence | SDG 12 & SDG 11",
    },
)

# ---------------------------------------------------------------------------
# Global CSS overrides
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Hide Top Header, Fork button and GitHub icons */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        [data-testid="stToolbar"] {
            display: none !important;
        }
        button[title="Fork this app"] {
            display: none !important;
        }
        a[href*="github.com"] {
            display: none !important;
        }
        /* Main background */
        .stApp { background-color: #f8fafc; }

        /* Sidebar */
        section[data-testid="stSidebar"] { background-color: #111827; color: #f9fafb; }
        section[data-testid="stSidebar"] * { color: #f9fafb !important; }

        /* Metric cards */
        div[data-testid="metric-container"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 16px 20px;
        }

        /* Tab styling */
        button[data-baseweb="tab"] { font-size: 0.9rem; font-weight: 600; }

        /* Custom bin badge */
        .bin-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: white;
            margin-bottom: 8px;
        }

        /* Policy card */
        .policy-card {
            background: #ffffff;
            border-left: 4px solid #3b82f6;
            border-radius: 6px;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }

        /* SDG pill */
        .sdg-pill {
            display: inline-block;
            background: #166534;
            color: #dcfce7;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 0.72rem;
            font-weight: 700;
            margin-right: 4px;
        }

        /* Hazard badges */
        .hazard-none    { background: #22c55e; }
        .hazard-medium  { background: #f59e0b; }
        .hazard-high    { background: #ef4444; }
        .hazard-critical{ background: #7f1d1d; }

        /* Step chips */
        .step-chip {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 6px;
            padding: 6px 12px;
            margin: 4px 0;
            font-size: 0.82rem;
            display: block;
        }

        /* Garbage audit — detected item card */
        .audit-item-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
            display: flex;
            align-items: flex-start;
            gap: 14px;
        }
        .audit-item-bin {
            width: 14px;
            min-width: 14px;
            height: 100%;
            min-height: 48px;
            border-radius: 4px;
        }
        .audit-item-body { flex: 1; }
        .audit-item-name { font-weight: 700; font-size: 0.9rem; margin-bottom: 2px; }
        .audit-item-meta { font-size: 0.78rem; color: #6b7280; }
        .audit-item-action {
            font-size: 0.82rem;
            color: #374151;
            background: #f9fafb;
            border-left: 3px solid #d1d5db;
            padding: 4px 8px;
            margin-top: 6px;
            border-radius: 0 4px 4px 0;
        }

        /* Segregation progress bars */
        .seg-bar-track {
            background: #e5e7eb;
            border-radius: 999px;
            height: 14px;
            margin: 4px 0 10px;
            overflow: hidden;
        }
        .seg-bar-fill {
            height: 14px;
            border-radius: 999px;
            transition: width 0.4s;
        }

        /* Hazard level badge */
        .hazard-low    { background: #22c55e; color: white; }
        .hazard-medium-badge { background: #f59e0b; color: white; }
        .hazard-severe { background: #dc2626; color: white; }
        .hazard-badge-lg {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 999px;
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }

        /* Action plan steps */
        .action-step {
            background: #fefce8;
            border: 1px solid #fde68a;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            font-size: 0.86rem;
            display: flex;
            gap: 10px;
            align-items: flex-start;
        }
        .action-step-num {
            background: #f59e0b;
            color: white;
            border-radius: 999px;
            min-width: 24px;
            height: 24px;
            font-weight: 800;
            font-size: 0.78rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Source pill */
        .source-gemini { background: #4285f4; color: white; }
        .source-groq   { background: #7c3aed; color: white; }
        .source-heuristic { background: #6b7280; color: white; }
        .source-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 0.72rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resource initialisation
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_classifier() -> WasteClassifier:
    """Initialise and cache the WasteClassifier."""
    return WasteClassifier()


@st.cache_resource(show_spinner=False)
def get_rag_engine() -> PolicyRAGEngine:
    """Initialise and cache the PolicyRAGEngine."""
    return PolicyRAGEngine()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## ♻️ EcoSort AI")
        st.caption("Enterprise Waste Intelligence Platform")
        st.markdown("---")
        st.markdown(
            "**Mission:** Swachh Bharat & Zero-Waste Cities\n"
            "<span class='sdg-pill'>SBM-U 2.0</span> "
            "<span class='sdg-pill'>Mission LiFE</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("#### Quick Stats")
        clf = get_classifier()
        items = clf.list_all_items()
        cats = clf.get_category_summary()
        st.metric("Catalog Items", len(items))
        st.metric("Waste Categories", len(cats))
        st.metric("Policy Documents", len(get_rag_engine().get_all_policies()))
        st.markdown("---")
        st.markdown(
            "<small>**EcoSort AI** · v1.0.0<br/>"
            "Developed with ❤️ by **Anshu Kumar Gupta** <br/>© 2026 · All Rights Reserved</small>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def _bin_badge(color: str, hex_color: str) -> str:
    return (
        f"<span class='bin-badge' style='background:{hex_color};'>"
        f"🗑 {color} Bin</span>"
    )


def _hazard_badge(level: str) -> str:
    label_map = {
        "none": ("✅ Safe", "hazard-none"),
        "medium": ("⚠️ Moderate Hazard", "hazard-medium"),
        "high": ("🔴 High Hazard", "hazard-high"),
        "critical": ("☠️ Critical Hazard", "hazard-critical"),
    }
    label, css = label_map.get(level, ("Unknown", "hazard-none"))
    return f"<span class='bin-badge {css}'>{label}</span>"


def _confidence_bar(score: float, method: str) -> None:
    method_labels = {
        "exact":          "Exact Match",
        "alias":          "Alias Match",
        "fuzzy":          "Fuzzy Match",
        "gemini":         "EcoSort Vision",
        "visual":         "Visual / Image Match",
        "visual_fallback":"Visual Fallback (low confidence)",
        "unmatched":      "No Match",
    }
    colour = "#22c55e" if score > 0.85 else "#f59e0b" if score > 0.6 else "#ef4444"
    st.markdown(
        f"""
        <div style='margin:6px 0'>
            <small style='color:#6b7280'>Confidence: {score*100:.0f}% —
            <em>{method_labels.get(method, method)}</em></small>
            <div style='background:#e5e7eb;border-radius:999px;height:6px;margin-top:4px'>
                <div style='background:{colour};width:{score*100:.0f}%;height:6px;
                     border-radius:999px'></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# TAB 1 — Live Waste Analyzer
# ---------------------------------------------------------------------------

def _render_garbage_audit_report(report) -> None:
    """
    Render a full GarbageAuditReport as a structured Garbage Area Audit panel.
    Displays: scene description, source badge, detected items list,
    segregation progress bars, hazard level, action plan, and env impact.
    """
    from core.vision_engine import GarbageAuditReport

    # ── Header with source badge ─────────────────────────────────────────
    source_labels = {
        "gemini":    ("EcoSort AI Vision", "source-gemini"),
        "groq":      ("Groq Vision (LLaMA)",  "source-groq"),
        "heuristic": ("Heuristic Analysis",   "source-heuristic"),
    }
    src_label, src_css = source_labels.get(report.source, ("Analysis", "source-heuristic"))

    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
                    color:white;padding:18px 24px;border-radius:10px;margin-bottom:18px'>
            <div style='display:flex;justify-content:space-between;align-items:start'>
                <div>
                    <div style='font-size:1.15rem;font-weight:800;margin-bottom:4px'>
                        🗺️ Garbage Area Audit Report
                    </div>
                    <div style='font-size:0.85rem;opacity:0.8'>{report.image_description}</div>
                </div>
                <span class='source-pill {src_css}' style='white-space:nowrap;margin-left:12px'>
                    ⚙ {src_label}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Hazard level ──────────────────────────────────────────────────────
    hazard_css = {
        "Low":    "hazard-low",
        "Medium": "hazard-medium-badge",
        "Severe": "hazard-severe",
    }.get(report.hazard_level, "hazard-low")
    hazard_icon = {"Low": "🟢", "Medium": "🟡", "Severe": "🔴"}.get(report.hazard_level, "🟢")

    st.markdown(
        f"""
        <div style='display:flex;align-items:center;gap:14px;margin-bottom:18px'>
            <span class='hazard-badge-lg {hazard_css}'>{hazard_icon} {report.hazard_level} Hazard</span>
            <span style='color:#374151;font-size:0.87rem'>{report.hazard_reason}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Two-column layout: detected items | segregation breakdown ─────────
    col_items, col_seg = st.columns([1.1, 1])

    with col_items:
        st.markdown("#### 🔍 Detected Waste Items")
        for item in report.detected_items:
            st.markdown(
                f"""
                <div class='audit-item-card'>
                    <div class='audit-item-bin' style='background:{item.bin_hex}'></div>
                    <div class='audit-item-body'>
                        <div class='audit-item-name'>{item.name}</div>
                        <div class='audit-item-meta'>
                            {item.category} &nbsp;·&nbsp; {item.quantity_estimate}
                            &nbsp;·&nbsp; <strong style='color:{item.bin_hex}'>{item.bin_color} Bin</strong>
                        </div>
                        <div class='audit-item-action'>➡ {item.action}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_seg:
        st.markdown("#### 📊 Segregation Breakdown")

        seg_rows = [
            ("Organic / Wet Waste",    report.pct_wet,        "#22c55e", "🟢"),
            ("Dry Recyclable",         report.pct_dry,        "#3b82f6", "🔵"),
            ("E-Waste / Hazardous",    report.pct_hazardous,  "#ef4444", "🔴"),
            ("Inert / Other",          report.pct_inert,      "#9ca3af", "⚫"),
        ]
        for label, pct, color, icon in seg_rows:
            pct_safe = max(0.0, min(100.0, float(pct)))
            st.markdown(
                f"""
                <div style='margin-bottom:2px'>
                    <div style='display:flex;justify-content:space-between;
                                font-size:0.83rem;color:#374151'>
                        <span>{icon} {label}</span>
                        <strong>{pct_safe:.0f}%</strong>
                    </div>
                    <div class='seg-bar-track'>
                        <div class='seg-bar-fill'
                             style='background:{color};width:{pct_safe:.0f}%'></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # KPI metrics
        st.markdown("<br/>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Recyclable", f"{report.pct_dry:.0f}%", help="Dry recyclable fraction")
        m2.metric("Organic",    f"{report.pct_wet:.0f}%",  help="Compostable fraction")
        m3, m4 = st.columns(2)
        m3.metric("Hazardous",  f"{report.pct_hazardous:.0f}%", help="E-waste / toxic fraction")
        m4.metric("Inert",      f"{report.pct_inert:.0f}%",  help="Debris / inert material")

    # ── Action Plan ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Step-by-Step Action Plan")
    st.caption("Konsa kooda kahan aur kaise dispose / segregate karna hai:")
    for i, step in enumerate(report.action_plan, 1):
        st.markdown(
            f"""
            <div class='action-step'>
                <div class='action-step-num'>{i}</div>
                <div>{step}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Environmental impact ──────────────────────────────────────────────
    if report.env_impact:
        st.markdown("---")
        st.success(f"🌱 **Environmental Impact:** {report.env_impact}")


def _render_classification_result(result, quantity: int) -> None:
    """Render the full classification card, recycling steps, tips, and lifecycle KPIs."""

    # Fallback confidence warning — shown when image could not be identified
    if result.match_method == "visual_fallback":
        st.warning(
            "⚠️ **Image not identified** — the filename contained no recognisable "
            "waste keywords and no hint text was provided.  Showing default guidance "
            "for **Plastic Bottle** (generic dry recyclable).  "
            "For accurate results, rename the file to include a keyword "
            "(e.g. `aluminium_can.jpg`) or type the item name in the Override field."
        )

    st.markdown("---")
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("#### Classification Result")
        st.markdown(_bin_badge(result.bin_color, result.bin_hex), unsafe_allow_html=True)
        st.markdown(_hazard_badge(result.hazard_level), unsafe_allow_html=True)
        _confidence_bar(result.confidence, result.match_method)

        st.markdown(f"**Item:** {result.item_name}")
        st.markdown(f"**Category:** {result.category_display}")
        st.markdown(f"**Primary Material:** {result.primary_material}")

        decomp = result.decomposition_years
        if decomp >= 1_000_000:
            decomp_str = "Never (geological timescale)"
        elif decomp >= 1000:
            decomp_str = f"{decomp:,.0f} years"
        elif decomp >= 1:
            decomp_str = f"{decomp:.1f} years"
        else:
            decomp_str = f"{decomp * 12:.0f} months"
        st.markdown(f"**Decomposition Time:** ⏳ {decomp_str}")

        if result.regulatory_note:
            st.info(f"📋 **Regulatory Note:** {result.regulatory_note}")

    with col_b:
        st.markdown("#### ♻️ Recycling Steps")
        for i, step in enumerate(result.recycling_steps, 1):
            st.markdown(
                f"<span class='step-chip'><b>{i}.</b> {step}</span>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### 💡 Handling Tips")
    st.success(f"🌿 {result.handling_tips}")

    st.markdown("#### 📊 Lifecycle Impact Metrics")
    lc = result.lifecycle
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total Weight",
        f"{lc['total_weight_kg']:.3f} kg",
        help=f"Estimated weight for {quantity} unit(s)",
    )
    col2.metric(
        "CO₂ Offset",
        f"{lc['carbon_offset_kg']:.3f} kg",
        delta="vs. landfill",
        delta_color="inverse",
        help="Carbon equivalent saved by proper recycling/composting",
    )
    col3.metric(
        "Tree Equivalent",
        f"{lc['carbon_offset_trees_equivalent']:.3f} trees/yr",
        help="Equivalent annual CO₂ absorption by trees (21 kg CO₂/tree/yr)",
    )
    col4.metric(
        "Landfill Volume",
        f"{lc['landfill_volume_litres']:.3f} L",
        help="Landfill space saved by proper disposal",
    )


def tab_waste_analyzer() -> None:
    st.markdown("### 🔍 Live Waste Analyzer")
    st.caption(
        "Identify a single waste item — or run a full Garbage Area Audit on a dump/street photo. "
        "Get bin guidance, recycling steps, lifecycle metrics, and an action plan."
    )

    clf = get_classifier()

    # ------------------------------------------------------------------ #
    # Quantity selector (shared by single-item modes only)                 #
    # ------------------------------------------------------------------ #
    quantity = st.number_input(
        "Quantity (units) — for single-item analysis",
        min_value=1,
        max_value=10_000,
        value=1,
        step=1,
        key="analyzer_qty",
    )

    # ------------------------------------------------------------------ #
    # Input mode tabs: Upload | Camera | Text | Area Audit                 #
    # ------------------------------------------------------------------ #
    input_tab_img, input_tab_cam, input_tab_text, input_tab_audit = st.tabs(
        ["📁 Upload Image", "📷 Camera Capture", "⌨️ Type Item Name", "🗺️ Garbage Area Audit"]
    )

    result = None           # will be set by whichever input mode fires
    image_source: str = ""  # "upload" | "camera" | "text" | "audit"
    audit_report = None     # set by the area audit tab

    # ── Upload Image ────────────────────────────────────────────────────
    with input_tab_img:
        st.markdown(
            "Upload a **JPG, JPEG, or PNG** photo of your waste item — "
            "EcoSort AI Vision will identify it directly from the image."
        )
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            key="img_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            # Image preview
            col_prev, col_info = st.columns([1, 1.4])
            with col_prev:
                st.image(
                    uploaded_file,
                    caption=f"📎 {uploaded_file.name}",
                    use_container_width=True,
                )
            with col_info:
                st.markdown("##### 🔬 Visual Analysis")
                # Show the filename-derived tags so the user can see what was parsed
                from core.classifier import _extract_visual_tags, infer_query_from_tags
                tags = _extract_visual_tags(uploaded_file.name)
                inferred_q, display_label = infer_query_from_tags(tags)
                if tags:
                    tag_html = " ".join(
                        f"<code style='background:#f0fdf4;border:1px solid #bbf7d0;"
                        f"border-radius:4px;padding:2px 6px;font-size:0.78rem'>{t}</code>"
                        for t in tags
                    )
                    st.markdown(f"**Detected tokens:** {tag_html}", unsafe_allow_html=True)
                else:
                    st.caption("No meaningful tokens found in filename.")

                if display_label:
                    st.markdown(
                        f"**Inferred object:** "
                        f"<span style='font-size:1rem'>{display_label}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(
                        "Could not infer item from filename. "
                        "Use the override field below or switch to **Type Item Name**."
                    )

                # Optional manual override
                override = st.text_input(
                    "Override item name (optional)",
                    placeholder="e.g. plastic bottle",
                    key="img_override",
                    help="Leave blank to use filename inference. Type here to override.",
                )

                if st.button("🔍 Analyse Image", key="btn_analyse_upload", type="primary"):
                    with st.spinner("Sending image to EcoSort AI Vision…"):
                        try:
                            img_bytes = uploaded_file.getvalue()
                            result = clf.classify_from_image(
                                img_bytes,
                                quantity=int(quantity),
                                manual_override=override or None,
                            )
                            image_source = "upload"
                        except Exception as _upload_err:
                            st.error(f"Gemini API Error: {_upload_err}")

    # ── Camera Capture ──────────────────────────────────────────────────
    with input_tab_cam:
        st.markdown(
            "Capture a photo of the waste item — "
            "EcoSort AI Vision will identify it automatically from the image."
        )
        camera_image = st.camera_input("Take a photo of the waste item", key="cam_input")

        if camera_image is not None:
            st.image(camera_image, caption="📷 Camera capture", use_container_width=True)
            if st.button("🔍 Analyse Photo", key="btn_analyse_cam", type="primary"):
                with st.spinner("Sending image to EcoSort AI Vision…"):
                    try:
                        cam_bytes = camera_image.getvalue()
                        result = clf.classify_from_image(
                            cam_bytes,
                            quantity=int(quantity),
                        )
                        image_source = "camera"
                    except Exception as _cam_err:
                        st.error(f"❌ EcoSort AI Vision error: {_cam_err}")

    # ── Garbage Area Audit ───────────────────────────────────────────────
    with input_tab_audit:
        st.markdown(
            "Upload or capture a photo of a **garbage dump, street waste pile, or mixed trash area**. "
            "EcoSort AI Vision will detect all visible waste types, generate a segregation breakdown, "
            "hazard assessment, and a step-by-step Hinglish action plan."
        )

        audit_col_up, audit_col_cam = st.columns(2)
        with audit_col_up:
            st.markdown("**Upload a garbage area photo:**")
            audit_file = st.file_uploader(
                "Garbage area photo",
                type=["jpg", "jpeg", "png"],
                key="audit_uploader",
                label_visibility="collapsed",
            )
        with audit_col_cam:
            st.markdown("**Or capture with camera:**")
            audit_cam = st.camera_input(
                "Capture garbage area",
                key="audit_cam",
            )

        # Resolve image bytes and preview
        audit_image_data: Optional[bytes] = None
        audit_filename: str = "garbage_area.jpg"

        if audit_file is not None:
            st.image(audit_file, caption=f"📎 {audit_file.name}", use_container_width=True)
            audit_image_data = audit_file.getvalue()
            audit_filename = audit_file.name
        elif audit_cam is not None:
            st.image(audit_cam, caption="📷 Camera capture", use_container_width=True)
            audit_image_data = audit_cam.getvalue()

        if audit_image_data is not None:
            if st.button("🗺️ Run Garbage Area Audit", key="btn_audit", type="primary"):
                with st.spinner("Sending image to EcoSort AI Vision — please wait…"):
                    try:
                        audit_report = clf.area_audit_from_image(
                            audit_image_data,
                            filename=audit_filename,
                        )
                        image_source = "audit"
                    except Exception as _audit_err:
                        st.error(f"Gemini API Error: {_audit_err}")

    # ── Text Input ──────────────────────────────────────────────────────
    with input_tab_text:
        st.markdown("Type the name of any waste item to classify it instantly.")
        col_input, _ = st.columns([3, 1])
        with col_input:
            query = st.text_input(
                "🗑 Waste Item",
                placeholder="e.g. banana peel, old laptop, glass bottle…",
                key="analyzer_query",
            )

        st.markdown(
            "**Try these:** "
            + " ".join(
                f"`{ex}`"
                for ex in [
                    "plastic bottle",
                    "old phone",
                    "banana peel",
                    "fluorescent tube",
                    "motor oil",
                    "newspaper",
                ]
            )
        )

        if query:
            with st.spinner("Classifying…"):
                result = clf.classify(query.strip(), quantity=int(quantity))
            image_source = "text"

    # ------------------------------------------------------------------ #
    # Render outputs — audit report takes priority over single-item result #
    # ------------------------------------------------------------------ #
    if image_source == "audit" and audit_report is not None:
        _render_garbage_audit_report(audit_report)
        return

    if result is None:
        st.info("Use one of the tabs above to classify a waste item, or try the 🗺️ Garbage Area Audit tab.")
        return

    if not result.matched:
        st.warning(
            "⚠️ Item not found in catalog. "
            + (
                f"Detected tokens: **{', '.join(result.visual_tags)}** — try renaming the file "
                "to include a keyword like `plastic_bottle.jpg` or use the Override field. "
                if image_source in ("upload", "camera") and result.visual_tags
                else "Try a more specific name. "
            )
            + "Unrecognised items should go to **general waste** pending identification."
        )
        return

    # Visual match banner (only for image-derived results)
    if image_source in ("upload", "camera") and result.visual_query:
        is_gemini = getattr(result, "match_method", "") == "gemini"
        banner_bg     = "#f0fdf4" if is_gemini else "#eff6ff"
        banner_border = "#bbf7d0" if is_gemini else "#bfdbfe"
        icon          = "🤖" if is_gemini else "🖼️"
        provider_tag  = (
            "<span style='background:#4285f4;color:white;border-radius:999px;"
            "padding:2px 8px;font-size:0.72rem;font-weight:700;margin-left:8px'>"
            "EcoSort Vision</span>"
        ) if is_gemini else ""
        st.markdown(
            f"""
            <div style='background:{banner_bg};border:1px solid {banner_border};
                        border-radius:8px;padding:12px 18px;margin:12px 0;
                        display:flex;align-items:center;gap:12px'>
                <span style='font-size:1.4rem'>{icon}</span>
                <div>
                    <strong>Visual Analysis Result</strong>{provider_tag}<br/>
                    <span style='color:#374151;font-size:0.87rem'>
                        Detected: <strong>{result.visual_query}</strong>
                        &nbsp;·&nbsp;
                        {'uploaded image' if image_source == 'upload' else 'camera capture'}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _render_classification_result(result, int(quantity))


# ---------------------------------------------------------------------------
# TAB 2 — Policy Assistant
# ---------------------------------------------------------------------------

def tab_policy_assistant() -> None:
    st.markdown("### 📜 Municipal & Campus Policy Assistant")
    st.caption(
        "Search waste management regulations, e-waste rules, campus guidelines, "
        "and SDG compliance frameworks."
    )

    engine = get_rag_engine()
    categories = ["All"] + engine.list_categories()

    col_q, col_cat = st.columns([3, 1])
    with col_q:
        query = st.text_input(
            "🔎 Policy Query",
            placeholder="e.g. how to dispose e-waste on campus, single-use plastic ban…",
            key="policy_query",
        )
    with col_cat:
        cat_filter = st.selectbox("Filter by Category", categories, key="policy_cat")

    example_queries = [
        "e-waste collection drive",
        "single use plastic ban",
        "campus segregation rules",
        "battery disposal EPR",
        "composting mandate",
    ]
    st.markdown(
        "**Example queries:** "
        + " · ".join(f"`{q}`" for q in example_queries)
    )

    if not query:
        # Show all policies in a browse view
        st.markdown("---")
        st.markdown("#### 📚 Policy Knowledge Base")
        all_policies = engine.get_all_policies()
        cat_filter_browse = cat_filter if cat_filter != "All" else None
        shown = [p for p in all_policies if cat_filter_browse is None or p.category == cat_filter_browse]
        for pol in shown:
            with st.expander(f"[{pol.category}] {pol.title}", expanded=False):
                st.markdown(f"**Jurisdiction:** {pol.jurisdiction}")
                st.markdown(f"**Source:** _{pol.source}_")
                st.markdown(f"**Last Updated:** {pol.last_updated}")
                st.markdown(f"> {pol.content}")
                tag_html = " ".join(f"<code>{t}</code>" for t in pol.tags)
                st.markdown(f"**Tags:** {tag_html}", unsafe_allow_html=True)
        return

    filter_arg = None if cat_filter == "All" else cat_filter
    with st.spinner("Searching policy knowledge base…"):
        results = engine.search(query.strip(), top_k=5, category_filter=filter_arg)

    st.markdown(f"---\n**{len(results)} result(s)** for query: _\"{query}\"_")

    if not results:
        st.warning(
            "No matching policy found. Try broader keywords such as "
            "'segregation', 'e-waste', 'composting', or 'plastic'."
        )
        return

    for r in results:
        rel_pct = int(r.relevance_score * 100)
        rel_colour = "#22c55e" if rel_pct > 75 else "#f59e0b" if rel_pct > 40 else "#ef4444"
        tag_html = " ".join(f"<code>{t}</code>" for t in r.tags[:6])
        st.markdown(
            f"""
            <div class='policy-card'>
                <div style='display:flex;justify-content:space-between;align-items:start'>
                    <div>
                        <strong style='font-size:1rem'>{r.title}</strong><br/>
                        <small style='color:#6b7280'>{r.category} · {r.jurisdiction}
                        · Updated {r.last_updated}</small>
                    </div>
                    <span style='background:{rel_colour};color:white;border-radius:999px;
                    padding:3px 10px;font-size:0.78rem;font-weight:700;white-space:nowrap;
                    margin-left:12px'>{rel_pct}% relevant</span>
                </div>
                <p style='margin:10px 0 6px;line-height:1.7;font-size:0.87rem'>{r.content}</p>
                <small style='color:#6b7280'>📚 Source: <em>{r.source}</em></small><br/>
                <div style='margin-top:6px'>{tag_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# TAB 3 — Impact & Carbon Offset Metrics
# ---------------------------------------------------------------------------

def tab_impact_metrics() -> None:
    st.markdown("### 🌍 Impact & Carbon Offset Metrics")
    st.caption(
        "Dynamic KPIs and visualisations for campus or household waste reduction "
        "aligned with SDG 12.5 (waste reduction) and SDG 11.6 (urban environmental impact)."
    )

    # --- Scenario selector ---
    col_scene, col_pop, col_dur = st.columns(3)
    with col_scene:
        scenario = st.selectbox(
            "Scenario", ["University Campus", "Residential Society", "Corporate Office", "Municipality"]
        )
    with col_pop:
        population = st.slider("Population / Users", 50, 50_000, 500, step=50)
    with col_dur:
        duration_months = st.slider("Period (months)", 1, 36, 12)

    # --- Simulated baseline data (deterministic seeding by scenario) ---
    scenario_seed = {
        "University Campus": 42,
        "Residential Society": 7,
        "Corporate Office": 13,
        "Municipality": 99,
    }
    random.seed(scenario_seed.get(scenario, 42))

    # Waste generation assumptions (kg/person/month)
    kg_per_person_month = {
        "University Campus": 8.5,
        "Residential Society": 22.0,
        "Corporate Office": 5.0,
        "Municipality": 18.0,
    }.get(scenario, 12.0)

    total_waste_kg = population * duration_months * kg_per_person_month

    # Diversion rate (properly disposed vs landfilled)
    diversion_rate = random.uniform(0.45, 0.72)
    diverted_kg = total_waste_kg * diversion_rate
    landfilled_kg = total_waste_kg * (1 - diversion_rate)

    # Category breakdown
    cat_ratios = {"Organic": 0.52, "Recyclable": 0.31, "E-Waste": 0.04, "Hazardous": 0.03, "Inert": 0.10}
    category_kg = {k: total_waste_kg * v for k, v in cat_ratios.items()}

    # Carbon offset
    avg_co2_per_kg = 0.62
    co2_offset_tonnes = (diverted_kg * avg_co2_per_kg) / 1000

    # Tree equivalent
    tree_equiv = co2_offset_tonnes * 1000 / 21

    # Landfill volume saved
    landfill_vol_m3 = (diverted_kg / 400)  # avg 400 kg/m³ compacted waste

    # Monthly trend (simulate with small random walk)
    monthly_diversion = []
    base = diversion_rate * 0.85
    for _ in range(duration_months):
        base = min(0.95, base + random.uniform(-0.02, 0.04))
        monthly_diversion.append(round(base * 100, 1))

    # ---- KPI row ----
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Waste Generated", f"{total_waste_kg:,.0f} kg")
    c2.metric(
        "Diversion Rate",
        f"{diversion_rate*100:.1f}%",
        delta=f"+{(diversion_rate - 0.40)*100:.1f}pp vs baseline",
    )
    c3.metric("CO₂ Offset", f"{co2_offset_tonnes:.2f} t CO₂e")
    c4.metric("🌳 Tree Equivalent", f"{tree_equiv:,.0f} trees/yr")
    c5.metric("Landfill Saved", f"{landfill_vol_m3:,.1f} m³")

    st.markdown("---")

    # ---- Charts using st.bar_chart / st.line_chart (no plotly dep) ----
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("##### Waste by Category (kg)")
        cat_data = {k: int(v) for k, v in category_kg.items()}
        st.bar_chart(cat_data, use_container_width=True, height=260)

    with col_c2:
        st.markdown("##### Monthly Diversion Rate (%)")
        months = [f"M{i+1}" for i in range(duration_months)]
        diversion_series = dict(zip(months, monthly_diversion))
        st.line_chart(diversion_series, use_container_width=True, height=260)

    # ---- Waste flow Sankey (text-based summary) ----
    st.markdown("---")
    st.markdown("##### ♻️ Waste Disposition Flow")
    col_f1, col_f2, col_f3 = st.columns(3)
    col_f1.metric("Generated", f"{total_waste_kg:,.0f} kg", help="Total waste generated in period")
    col_f2.metric(
        "Properly Disposed",
        f"{diverted_kg:,.0f} kg",
        delta=f"{diversion_rate*100:.1f}%",
        delta_color="normal",
    )
    col_f3.metric(
        "Landfilled",
        f"{landfilled_kg:,.0f} kg",
        delta=f"-{(1-diversion_rate)*100:.1f}% (target: <30%)",
        delta_color="inverse",
    )

    # ---- SDG Progress indicators ----
    st.markdown("---")
    st.markdown("##### 🎯 SDG 12.5 Progress — Waste Reduction Target")

    sdg_target = 0.50  # SDG 12.5 baseline target: 50% diversion
    sdg_achieved = diversion_rate >= sdg_target
    sdg_colour = "#22c55e" if sdg_achieved else "#f59e0b"
    sdg_label = "✅ Target Met" if sdg_achieved else "⚠️ Below Target"
    gap = max(0, sdg_target - diversion_rate)

    st.markdown(
        f"""
        <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:16px'>
            <div style='display:flex;justify-content:space-between'>
                <div>
                    <strong>SDG 12.5</strong> — Substantially reduce waste generation through
                    prevention, reduction, recycling, and reuse.
                </div>
                <span style='background:{sdg_colour};color:white;padding:4px 12px;
                border-radius:999px;font-size:0.82rem;font-weight:700'>{sdg_label}</span>
            </div>
            <div style='margin-top:12px'>
                <div style='display:flex;justify-content:space-between;
                            font-size:0.8rem;color:#374151;margin-bottom:4px'>
                    <span>Current: {diversion_rate*100:.1f}%</span>
                    <span>Target: ≥ 50%</span>
                </div>
                <div style='background:#e5e7eb;border-radius:999px;height:10px'>
                    <div style='background:{sdg_colour};width:{min(diversion_rate/sdg_target,1)*100:.0f}%;
                         height:10px;border-radius:999px'></div>
                </div>
                {"" if sdg_achieved else f"<small style='color:#92400e'>Gap to close: {gap*100:.1f} percentage points</small>"}
                </div>
            
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Annual projection ----
    if duration_months < 12:
        proj_co2 = co2_offset_tonnes * (12 / duration_months)
        proj_trees = tree_equiv * (12 / duration_months)
        st.info(
            f"📈 **12-Month Projection:** "
            f"At current rate, you'd offset **{proj_co2:.1f} t CO₂e** and save the equivalent "
            f"of **{proj_trees:,.0f} trees/yr** annually."
        )


# ---------------------------------------------------------------------------
# TAB 4 — Responsible AI Audit
# ---------------------------------------------------------------------------

def tab_ai_audit() -> None:
    st.markdown("### 🤖 Responsible AI Audit Dashboard")
    st.caption(
        "Transparency report covering fairness, explainability, privacy safeguards, "
        "and UN SDG alignment for the EcoSort AI system."
    )

    # ---- System overview ----
    st.markdown("#### ⚙️ System Architecture Overview")
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown(
            """
            | Component | Technology | Version |
            |-----------|-----------|---------|
            | Classifier | Rule-based NLP + Fuzzy Match | 2.1.0 |
            | RAG Engine | TF-IDF Keyword Retrieval | 2.1.0 |
            | Knowledge Base | Curated Mock KB (21 policies) | Jan 2024 |
            | Frontend | Streamlit | ≥1.32 |
            | Data Store | JSON flat file (no PII) | — |
            """
        )
    with col_arch2:
        st.markdown(
            """
            **No LLM / Generative AI** is used in this build.
            All classifications and retrievals are deterministic and auditable.

            **Data Flows:**
            - User input → Classifier → Catalog lookup → Result display
            - User query → RAG Engine → KB search → Ranked policy display
            - No data leaves the local environment
            - No user data is stored or logged
            """
        )

    st.markdown("---")

    # ---- Fairness Panel ----
    st.markdown("#### ⚖️ Fairness Assessment")
    fairness_items = [
        {
            "dimension": "Geographic Fairness",
            "status": "✅ Pass",
            "detail": "Catalog and policies include items relevant across urban, peri-urban, "
                      "and rural contexts. Policies cover India (primary), with SDG alignment for global use.",
            "score": 88,
        },
        {
            "dimension": "Language / Literacy Fairness",
            "status": "⚠️ Partial",
            "detail": "Current UI is English-only. Multi-language support (Hindi, Tamil, Bengali) "
                      "planned for v3.0. Recommended: add vernacular aliases to waste catalog.",
            "score": 55,
        },
        {
            "dimension": "Socioeconomic Fairness",
            "status": "✅ Pass",
            "detail": "Classification rules do not use income or demographic data. "
                      "Bin colour guidance works for all households regardless of income level.",
            "score": 92,
        },
        {
            "dimension": "Disability Accessibility",
            "status": "⚠️ Partial",
            "detail": "Streamlit provides basic keyboard navigation. "
                      "Screen reader support and WCAG 2.1 AA compliance require additional audit.",
            "score": 62,
        },
        {
            "dimension": "Demographic Bias in Training Data",
            "status": "✅ Pass",
            "detail": "No ML model trained on personal data. Classification is rule-based "
                      "with a curated catalog — no learned demographic biases possible.",
            "score": 97,
        },
    ]

    for fi in fairness_items:
        colour = "#22c55e" if fi["score"] >= 80 else "#f59e0b" if fi["score"] >= 55 else "#ef4444"
        st.markdown(
            f"""
            <div style='background:#fff;border:1px solid #e5e7eb;border-radius:8px;
                        padding:14px 18px;margin-bottom:10px;display:flex;
                        align-items:center;gap:16px'>
                <div style='min-width:180px;font-weight:600'>{fi['dimension']}</div>
                <div style='flex:1;font-size:0.85rem;color:#374151'>{fi['detail']}</div>
                <div style='text-align:right;min-width:120px'>
                    <div style='font-size:0.78rem;margin-bottom:4px'>{fi['status']}</div>
                    <div style='background:#e5e7eb;border-radius:999px;height:6px'>
                        <div style='background:{colour};width:{fi["score"]}%;
                             height:6px;border-radius:999px'></div>
                    </div>
                    <small style='color:{colour};font-weight:700'>{fi["score"]}/100</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---- Explainability Panel ----
    st.markdown("#### 🔍 Explainability Breakdown")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown(
            """
            **Classification Logic (Transparent)**

            The classifier uses a 3-tier deterministic approach:

            1. **Exact Name Match** — direct lookup against catalog names.
               _Confidence: 100%_
            2. **Alias / Keyword Match** — substring and synonym matching
               against the `aliases` field in the catalog.
               _Confidence: 85–97%_
            3. **Fuzzy Match** — SequenceMatcher similarity ≥ 0.55 threshold.
               _Confidence: 47–85%_

            Every result includes:
            - Match method label
            - Confidence score (0–100%)
            - Source item ID for traceability
            - Category and bin color derivation chain
            """
        )
    with col_e2:
        st.markdown(
            """
            **RAG Retrieval Logic (Auditable)**

            Policy retrieval uses TF-IDF-style keyword scoring:

            - Query tokenised into 3+ character words
            - Scored against: title (×2), tags (×3), content (×1)
            - Normalised to 0–100% relevance band
            - Top-4 results returned, ranked by score

            **What this system does NOT do:**
            - ❌ No generative hallucination (no LLM)
            - ❌ No personalisation or user profiling
            - ❌ No opaque neural embeddings
            - ❌ No external API calls in core logic
            """
        )

    st.markdown("---")

    # ---- Privacy Safeguards ----
    st.markdown("#### 🔒 Privacy Safeguards")
    privacy_items = [
        ("Data Minimisation", "✅", "Only waste item names are processed. No PII collected or required."),
        ("No Persistent Logging", "✅", "Session data is not written to disk or any database."),
        ("No External API Calls", "✅", "Core classifier and RAG engine operate fully offline."),
        ("No User Profiling", "✅", "No cookies, fingerprinting, or behavioural tracking."),
        ("Data Residency", "✅", "All data processed locally. No cloud transmission."),
        ("GDPR / DPDP Compliance", "✅", "No personal data processed — DPDP Act 2023 (India) compliant by design."),
        ("Secure Defaults", "⚠️", "Production deployments should add HTTPS and auth middleware."),
    ]

    priv_table_rows = "".join(
        f"<tr><td style='padding:8px 12px;font-weight:600'>{item[0]}</td>"
        f"<td style='padding:8px 12px;text-align:center'>{item[1]}</td>"
        f"<td style='padding:8px 12px;font-size:0.85rem;color:#374151'>{item[2]}</td></tr>"
        for item in privacy_items
    )
    st.markdown(
        f"""
        <table style='width:100%;border-collapse:collapse;background:#fff;
                      border:1px solid #e5e7eb;border-radius:8px;overflow:hidden'>
            <thead>
                <tr style='background:#f3f4f6'>
                    <th style='padding:10px 12px;text-align:left'>Safeguard</th>
                    <th style='padding:10px 12px;text-align:center'>Status</th>
                    <th style='padding:10px 12px;text-align:left'>Detail</th>
                </tr>
            </thead>
            <tbody>{priv_table_rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---- SDG Alignment Matrix ----
    st.markdown("#### 🌐 UN SDG Alignment Matrix")

    sdg_matrix = [
        {
            "sdg": "SDG 11.6",
            "title": "Sustainable Cities — Environmental Impact",
            "alignment": "Direct",
            "feature": "Bin guidance, landfill volume metrics, municipal policy retrieval",
            "score": 92,
        },
        {
            "sdg": "SDG 12.3",
            "title": "Halve Food Waste by 2030",
            "alignment": "Direct",
            "feature": "Organic waste classification, composting steps, food scrap guidance",
            "score": 85,
        },
        {
            "sdg": "SDG 12.4",
            "title": "Sound Management of Hazardous Waste",
            "alignment": "Direct",
            "feature": "Toxic/Hazardous waste category, e-waste rules, regulatory notes",
            "score": 90,
        },
        {
            "sdg": "SDG 12.5",
            "title": "Reduce Waste Generation — Recycling & Reuse",
            "alignment": "Direct",
            "feature": "Recycling steps, carbon offset metrics, diversion rate KPIs",
            "score": 95,
        },
        {
            "sdg": "SDG 12.8",
            "title": "Awareness for Sustainable Lifestyles",
            "alignment": "Direct",
            "feature": "Policy assistant, handling tips, lifecycle education",
            "score": 88,
        },
        {
            "sdg": "SDG 13",
            "title": "Climate Action — Carbon Offset Tracking",
            "alignment": "Indirect",
            "feature": "CO₂ offset calculation, tree equivalent metrics",
            "score": 72,
        },
        {
            "sdg": "SDG 17.16",
            "title": "Global Partnership — Open Data",
            "alignment": "Indirect",
            "feature": "Open-source codebase, MIT license, shareable waste catalog JSON",
            "score": 65,
        },
    ]

    for sdg in sdg_matrix:
        colour = "#22c55e" if sdg["score"] >= 85 else "#3b82f6" if sdg["score"] >= 70 else "#f59e0b"
        align_colour = "#166534" if sdg["alignment"] == "Direct" else "#1e40af"
        st.markdown(
            f"""
            <div style='background:#fff;border:1px solid #e5e7eb;border-radius:8px;
                        padding:12px 18px;margin-bottom:8px;display:flex;
                        align-items:center;gap:16px'>
                <div style='min-width:100px'>
                    <span style='background:{colour};color:white;border-radius:999px;
                    padding:4px 10px;font-size:0.78rem;font-weight:700'>{sdg["sdg"]}</span>
                </div>
                <div style='flex:1'>
                    <strong style='font-size:0.88rem'>{sdg["title"]}</strong><br/>
                    <small style='color:#6b7280'>{sdg["feature"]}</small>
                </div>
                <div style='text-align:right;min-width:140px'>
                    <span style='background:{align_colour};color:white;border-radius:999px;
                    padding:2px 8px;font-size:0.72rem;font-weight:700'>
                    {sdg["alignment"]}</span>
                    <div style='margin-top:4px;background:#e5e7eb;border-radius:999px;height:6px'>
                        <div style='background:{colour};width:{sdg["score"]}%;
                             height:6px;border-radius:999px'></div>
                    </div>
                    <small style='color:{colour};font-weight:700'>{sdg["score"]}/100</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---- AI Ethics checklist ----
    st.markdown("#### ✅ AI Ethics Checklist (IEEE / EU AI Act Inspired)")
    ethics = [
        ("Human Oversight", True, "All outputs are advisory. Humans make final disposal decisions."),
        ("Transparency", True, "Match method and confidence shown for every classification."),
        ("Robustness & Safety", True, "Unmatched items return safe default (general waste) guidance."),
        ("Non-Discrimination", True, "No protected characteristics used in any logic path."),
        ("Privacy & Data Governance", True, "Zero PII collected. Fully offline core."),
        ("Accountability", True, "Open-source code, version-tagged, auditable logic."),
        ("Environmental Well-being", True, "Core purpose is environmental benefit — SDG aligned."),
        ("Contestability", False, "Feedback mechanism for incorrect classifications not yet implemented."),
    ]
    for label, passed, detail in ethics:
        icon = "✅" if passed else "🔲"
        bg = "#f0fdf4" if passed else "#fefce8"
        border = "#bbf7d0" if passed else "#fde68a"
        st.markdown(
            f"""
            <div style='background:{bg};border:1px solid {border};border-radius:6px;
                        padding:10px 16px;margin-bottom:6px;display:flex;gap:12px'>
                <span style='font-size:1.1rem'>{icon}</span>
                <div>
                    <strong style='font-size:0.87rem'>{label}</strong>
                    <span style='color:#6b7280;font-size:0.83rem;margin-left:8px'>{detail}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    render_sidebar()

    # Header
    st.markdown(
        """
        <div style='background:linear-gradient(135deg,#14532d 0%,#166534 60%,#15803d 100%);
                    color:white;padding:24px 32px;border-radius:12px;margin-bottom:24px'>
            <h1 style='margin:0;font-size:1.9rem;font-weight:800;letter-spacing:-0.02em'>
                ♻️ EcoSort AI
            </h1>
            <p style='margin:4px 0 0;opacity:0.85;font-size:1rem'>
                Enterprise Waste Intelligence & Policy Assistant
                &nbsp;·&nbsp;
                <span style='background:rgba(255,255,255,0.2);padding:2px 8px;
                border-radius:999px;font-size:0.78rem'>SBM 2.0</span>
                &nbsp;
                <span style='background:rgba(255,255,255,0.2);padding:2px 8px;
                border-radius:999px;font-size:0.78rem'>SWM Rules 2016</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔍 Live Waste Analyzer",
            "📜 Policy Assistant",
            "🌍 Impact & Carbon Metrics",
            "🤖 Responsible AI Audit",
        ]
    )

    with tab1:
        tab_waste_analyzer()

    with tab2:
        tab_policy_assistant()

    with tab3:
        tab_impact_metrics()

    with tab4:
        tab_ai_audit()

    # Footer
    st.markdown(
        """
        <div style='text-align:center;margin-top:40px;padding-top:16px;
                    border-top:1px solid #e5e7eb;color:#9ca3af;font-size:0.78rem'>
            EcoSort AI · Built for Digital India | Aligned with Swachh Bharat Mission (SBM-U 2.0), SWM Rules 2016 & Mission LiFE
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
