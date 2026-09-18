"""
EcoSort AI — Policy RAG Engine
================================
Retrieval-Augmented Generation (RAG) engine for municipal waste policies,
e-waste rules, and campus segregation guidelines.

Architecture:
  - A curated mock knowledge-base (KB) of policy documents is stored as
    structured dicts (simulating a vector-store retrieval without a live API).
  - TF-IDF-style keyword scoring retrieves the most relevant policy chunks.
  - Results are ranked by relevance score and returned with source metadata.

SDG Alignment: SDG 11 (Sustainable Cities), SDG 12 (Responsible Production)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    """A single retrieved policy chunk with relevance metadata."""

    title: str
    category: str          # e.g. "Municipal", "E-Waste", "Campus", "National"
    content: str
    source: str
    relevance_score: float
    tags: list[str] = field(default_factory=list)
    jurisdiction: str = "India (General)"
    last_updated: str = "2024-01"


# ---------------------------------------------------------------------------
# Mock knowledge base
# ---------------------------------------------------------------------------

_POLICY_KB: list[dict] = [
    # ---- Municipal solid waste ----
    {
        "id": "POL-M01",
        "title": "Source Segregation — Mandatory Dual Bin Rule",
        "category": "Municipal",
        "jurisdiction": "India (SWM Rules 2016)",
        "source": "Solid Waste Management Rules, 2016 — MoEFCC India",
        "last_updated": "2022-03",
        "tags": ["segregation", "dual bin", "green bin", "blue bin", "wet waste", "dry waste"],
        "content": (
            "As per the Solid Waste Management (SWM) Rules 2016 and their 2022 amendments, "
            "all households, bulk waste generators (BWG), and commercial establishments are "
            "MANDATED to segregate waste at source into minimum three categories: "
            "(1) Wet/Organic waste — green lidded bin, "
            "(2) Dry/Recyclable waste — blue lidded bin, "
            "(3) Domestic Hazardous & Sanitary waste — black/red bin. "
            "Unsegregated waste must NOT be accepted by collection vehicles. "
            "Non-compliance attracts spot fines of ₹500–₹5,000 depending on ULB by-laws. "
            "Bulk waste generators (>100 kg/day) must process wet waste on-site."
        ),
    },
    {
        "id": "POL-M02",
        "title": "Collection Schedule and Vehicle Timings",
        "category": "Municipal",
        "jurisdiction": "Urban Local Bodies — General",
        "source": "CPHEEO Manual on Municipal Solid Waste Management",
        "last_updated": "2023-06",
        "tags": ["collection", "schedule", "garbage truck", "pickup", "timing"],
        "content": (
            "Municipal collection vehicles operate on fixed schedules: "
            "Wet waste is collected daily or every alternate day. "
            "Dry/recyclable waste is typically collected twice a week. "
            "Domestic hazardous waste (batteries, CFLs, medicines) is collected "
            "on designated monthly drive days. "
            "Citizens must place bins at designated collection points by 7:00 AM. "
            "Late-out bins may be levied a ₹200 fine under ULB by-laws. "
            "Collection drives for e-waste are announced quarterly on ULB portals."
        ),
    },
    {
        "id": "POL-M03",
        "title": "Wet Waste — Composting and Biogas Mandate",
        "category": "Municipal",
        "jurisdiction": "India (SWM Rules 2016)",
        "source": "MoEFCC Solid Waste Management Rules 2016 — Rule 4(f)",
        "last_updated": "2022-05",
        "tags": ["composting", "biogas", "organic", "wet waste", "processing"],
        "content": (
            "Rule 4(f) of SWM Rules 2016 mandates that: "
            "Gated communities and bulk generators with >100 kg/day wet waste MUST "
            "set up on-site composting or biogas units. "
            "Compost produced must meet FSSAI/quality standards before land application. "
            "Municipal composting facilities must achieve >50% diversion from landfill by 2025. "
            "Biogas plants may qualify for carbon credits under CDM and VCS schemes. "
            "Food waste processors (IWC-type) are permitted for apartment complexes ≥ 50 units."
        ),
    },
    {
        "id": "POL-M04",
        "title": "Single-Use Plastic Ban Enforcement",
        "category": "Municipal",
        "jurisdiction": "India — National Ban (Jul 2022)",
        "source": "Environment Protection Act 1986 — Plastic Waste Management Amendment Rules 2022",
        "last_updated": "2022-07",
        "tags": ["plastic ban", "single use plastic", "SUP", "polythene", "straw", "cutlery"],
        "content": (
            "Effective 1 July 2022, India banned 19 categories of single-use plastics (SUP) "
            "including: plastic straws, cutlery, plates, cups, carry bags <75 microns, "
            "thermocol items, and candy sticks. "
            "Manufacturing, stocking, distribution, and sale are prohibited. "
            "Fines: ₹500–₹25,000 for individuals; ₹5 lakh–₹25 lakh for businesses. "
            "Alternatives: use cloth bags (>120 GSM), steel/bamboo cutlery, paper straws. "
            "Carry bags ≥75 microns remain permitted but must be sold for ≥₹1 per bag."
        ),
    },
    # ---- E-Waste ----
    {
        "id": "POL-E01",
        "title": "E-Waste Management Rules 2016 — Producer Responsibility",
        "category": "E-Waste",
        "jurisdiction": "India — CPCB / MoEFCC",
        "source": "E-Waste (Management) Rules 2016 — Amended 2022",
        "last_updated": "2023-01",
        "tags": ["e-waste", "producer responsibility", "EPR", "take-back", "CPCB", "electronics"],
        "content": (
            "The E-Waste (Management) Rules 2016 (amended 2022) establish Extended Producer "
            "Responsibility (EPR): Producers of 21 categories of electrical/electronic equipment "
            "MUST collect back and channel their products for authorised dismantling/recycling. "
            "EPR targets (% of sales weight): FY2023: 60%, FY2024: 70%, FY2025: 80%. "
            "Producers must register on CPCB's EPR portal and file annual returns. "
            "Retailers must set up or participate in take-back collection schemes. "
            "Consumers must NOT discard e-waste in municipal bins. "
            "Authorised dismantlers must be CPCB/State PCB certified."
        ),
    },
    {
        "id": "POL-E02",
        "title": "E-Waste Collection Drives — Campus and Corporate",
        "category": "E-Waste",
        "jurisdiction": "General — India",
        "source": "CPCB E-Waste Guidelines 2020 / Saahas Zero Waste Model",
        "last_updated": "2023-06",
        "tags": ["e-waste drive", "collection drive", "campus", "corporate", "kiosk"],
        "content": (
            "E-waste collection drives are the recommended mechanism for bulk collection: "
            "Corporates and educational institutions must conduct at least TWO e-waste drives "
            "per year as part of EPR obligations. "
            "Accepted items: computers, monitors, printers, phones, batteries, cables, peripherals. "
            "Items must be inventoried with serial numbers before handover to authorised recyclers. "
            "Certificates of Recycling (CoR) are issued by authorised processors and must be "
            "retained for 3 years for compliance audits. "
            "NGOs like Saahas, E-Parisaraa, and Attero facilitate campus e-waste drives. "
            "Drop-box kiosks can be installed at facility entrances — contact your PCB for list."
        ),
    },
    {
        "id": "POL-E03",
        "title": "Hazardous Waste — Battery Disposal",
        "category": "E-Waste",
        "jurisdiction": "India — Battery Waste Management Rules 2022",
        "source": "Battery Waste Management Rules 2022 — MoEFCC",
        "last_updated": "2023-01",
        "tags": ["battery", "lithium", "lead acid", "disposal", "recycling", "EPR"],
        "content": (
            "Battery Waste Management Rules 2022 replace earlier battery rules and extend "
            "EPR to all battery types: portable, automotive, industrial, and EV batteries. "
            "Producers must achieve collection efficiency targets: 70% by FY2026. "
            "Retailers must accept used batteries from consumers free of charge. "
            "Lithium-ion battery recycling must recover ≥95% of lithium, cobalt, nickel. "
            "Lead-acid batteries: ≥90% recycling efficiency mandatory. "
            "Illegal dumping of batteries attracts action under Environment Protection Act 1986. "
            "EPR credit trading mechanism notified — excess credits may be transferred between producers."
        ),
    },
    # ---- Campus / Institutional ----
    {
        "id": "POL-C01",
        "title": "Campus Waste Segregation — Green Campus Guidelines",
        "category": "Campus",
        "jurisdiction": "AICTE / UGC India",
        "source": "UGC Green Campus Initiative 2018 / AICTE Model Guidelines",
        "last_updated": "2022-09",
        "tags": ["campus", "university", "college", "segregation", "green campus", "hostel"],
        "content": (
            "UGC Green Campus Initiative and AICTE model guidelines mandate: "
            "All educational institutions must implement 4-bin segregation: "
            "Green (wet), Blue (dry/recyclable), Red (biomedical/e-waste), Black (hazardous). "
            "Campus canteens must reduce food waste by 20% annually and compost kitchen waste. "
            "Plastic-free campus policy: ban single-use plastics in canteens, events, and hostels. "
            "Annual waste audit report to be submitted to UGC/AICTE. "
            "NAAC and NBA accreditation now include campus sustainability as a parameter. "
            "Institutions with >500 students are classified as bulk generators under SWM Rules."
        ),
    },
    {
        "id": "POL-C02",
        "title": "Hostel and Dormitory Waste Management",
        "category": "Campus",
        "jurisdiction": "General — India",
        "source": "EcoSort AI Campus Best Practices Guideline v1.2",
        "last_updated": "2023-08",
        "tags": ["hostel", "dormitory", "student", "room waste", "mess"],
        "content": (
            "Best practices for hostel waste management: "
            "Each room must have two waste bags (wet + dry). "
            "Mess/canteen food waste must be composted daily using IWC or aerobic composting. "
            "Prohibited in hostel bins: batteries, CFLs, sharps, chemicals, e-waste. "
            "Students must participate in monthly e-waste collection drives. "
            "Waste wardens (student volunteers) to be appointed per floor for compliance. "
            "Paper and cardboard from hostels must be bundled monthly for paper drives. "
            "Night soil/sanitary waste (pads, diapers) must go to black/sanitary waste bags."
        ),
    },
    # ---- Medical / Biomedical ----
    {
        "id": "POL-B01",
        "title": "Biomedical Waste Management Rules 2016",
        "category": "Biomedical",
        "jurisdiction": "India — CPCB",
        "source": "Biomedical Waste Management Rules 2016 — MoEFCC",
        "last_updated": "2023-02",
        "tags": ["biomedical", "hospital", "medical waste", "sharps", "needle", "yellow bag", "red bag"],
        "content": (
            "Biomedical Waste Management Rules 2016 classify medical waste into 4 colour-coded categories: "
            "YELLOW bag — anatomical/pathological waste, pharmaceutical waste, chemical waste. "
            "RED bag — contaminated recyclable waste (IV sets, syringes without needles, tubes). "
            "WHITE/TRANSLUCENT puncture-proof container — sharps (needles, blades, lancets). "
            "BLUE bag — glassware, metallic body implants. "
            "Home-generated medical waste (insulin needles, lancets) must go to sharps containers "
            "and be deposited at local PHC/hospital collection points. "
            "Open burning of biomedical waste is strictly prohibited."
        ),
    },
    # ---- National/Global policies ----
    {
        "id": "POL-N01",
        "title": "Swachh Bharat Mission — Urban Waste Targets",
        "category": "National",
        "jurisdiction": "India — National Mission",
        "source": "Swachh Bharat Mission (Urban) Phase II, MoHUA",
        "last_updated": "2023-04",
        "tags": ["swachh bharat", "ODF", "open defecation", "clean city", "urban mission", "landfill"],
        "content": (
            "Swachh Bharat Mission Urban 2.0 (2021-2026) targets: "
            "All cities to achieve ODF++ and Water+ status. "
            "100% source segregation in all ULBs by 2026. "
            "Waste processing capacity to reach 100% of waste generated. "
            "Remediation of all legacy landfill sites (1800+ sites across India). "
            "All cities with population >1 lakh to achieve 3-star Garbage Free City rating. "
            "₹1.41 lakh crore allocated for urban cleanliness and solid waste infrastructure. "
            "Cities ranked annually on Swachh Survekshan — ratings tied to municipal funding."
        ),
    },
    {
        "id": "POL-N02",
        "title": "UN SDG 12 — Responsible Consumption and Production",
        "category": "Global",
        "jurisdiction": "United Nations",
        "source": "UN 2030 Agenda for Sustainable Development — Goal 12",
        "last_updated": "2023-09",
        "tags": ["SDG 12", "sustainable consumption", "circular economy", "waste reduction", "UN"],
        "content": (
            "UN SDG 12 targets relevant to waste management: "
            "12.3 — Halve per capita food waste at retail/consumer level by 2030. "
            "12.4 — Achieve environmentally sound management of chemicals and hazardous wastes. "
            "12.5 — Substantially reduce waste generation through prevention, reduction, recycling. "
            "12.6 — Encourage companies to adopt sustainable reporting practices. "
            "12.8 — Ensure all people have relevant information for sustainable lifestyles. "
            "Circular economy principles: design out waste, keep materials in use, regenerate systems. "
            "Extended Producer Responsibility (EPR) is the primary policy instrument for SDG 12.5."
        ),
    },
    {
        "id": "POL-N03",
        "title": "Construction and Demolition Waste Management Rules",
        "category": "Municipal",
        "jurisdiction": "India — MoEFCC",
        "source": "C&D Waste Management Rules 2016 — MoEFCC India",
        "last_updated": "2022-11",
        "tags": ["construction", "demolition", "C&D waste", "debris", "concrete", "bricks"],
        "content": (
            "Construction and Demolition Waste Management Rules 2016: "
            "C&D waste must be SEGREGATED into concrete, soil, metal, glass, and wood fractions. "
            "Generators (>20 tonnes or 300 sqft) must obtain waste management plan approval from ULB. "
            "C&D waste must NOT be mixed with municipal solid waste. "
            "Processed C&D aggregate can replace virgin material in road sub-base and fill. "
            "C&D recycling plants must be set up by ULBs in cities with >1 lakh population. "
            "Illegal dumping on roads, water bodies, and open plots attracts ₹5,000–₹50,000 fines."
        ),
    },
]


# ---------------------------------------------------------------------------
# TF-IDF-style keyword scorer
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Tokenise text into lowercase words (3+ chars)."""
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w]


def _score_document(query_tokens: list[str], doc: dict) -> float:
    """
    Simple TF-style scoring: count query token hits across title, tags, content.
    Title hits count double; tag exact-matches count triple.
    """
    title_tokens = _tokenise(doc["title"])
    tag_tokens = [t.lower().replace(" ", "") for t in doc.get("tags", [])]
    content_tokens = _tokenise(doc["content"])

    score = 0.0
    for qt in query_tokens:
        # Title hit (weight 2)
        score += 2.0 * title_tokens.count(qt)
        # Tag hit (weight 3 for exact tag match)
        for tag in tag_tokens:
            if qt in tag or tag in qt:
                score += 3.0
                break
        # Content hit (weight 1)
        score += 1.0 * content_tokens.count(qt)

    # Normalise by query length to get a per-token score
    if query_tokens:
        score /= len(query_tokens)
    return score


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class PolicyRAGEngine:
    """
    Retrieves the most relevant policy chunks for a free-text query.

    Usage::

        engine = PolicyRAGEngine()
        results = engine.search("how do I dispose e-waste on campus?")
        for r in results:
            print(r.title, r.relevance_score)
    """

    def __init__(self, kb: Optional[list[dict]] = None) -> None:
        """
        Parameters
        ----------
        kb : list[dict], optional
            Override the default knowledge base (useful for testing).
        """
        self._kb = kb if kb is not None else _POLICY_KB

    def search(
        self,
        query: str,
        top_k: int = 4,
        category_filter: Optional[str] = None,
    ) -> list[PolicyResult]:
        """
        Retrieve the top-k most relevant policy documents for a query.

        Parameters
        ----------
        query : str
            Natural language policy question.
        top_k : int
            Maximum number of results to return (default 4).
        category_filter : str, optional
            If set, restrict results to this category (e.g. "E-Waste").

        Returns
        -------
        list[PolicyResult]
            Ranked list of policy results, highest relevance first.
        """
        if not query or not query.strip():
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, dict]] = []
        for doc in self._kb:
            if category_filter and doc.get("category", "").lower() != category_filter.lower():
                continue
            score = _score_document(query_tokens, doc)
            if score > 0:
                scored.append((score, doc))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Cap relevance at 1.0 for display
        max_score = scored[0][0] if scored else 1.0

        results = []
        for score, doc in scored[:top_k]:
            results.append(
                PolicyResult(
                    title=doc["title"],
                    category=doc.get("category", "General"),
                    content=doc["content"],
                    source=doc.get("source", ""),
                    relevance_score=round(min(score / max_score, 1.0), 3),
                    tags=doc.get("tags", []),
                    jurisdiction=doc.get("jurisdiction", "India"),
                    last_updated=doc.get("last_updated", ""),
                )
            )

        return results

    def list_categories(self) -> list[str]:
        """Return unique policy categories in the knowledge base."""
        cats = sorted({doc.get("category", "General") for doc in self._kb})
        return cats

    def get_all_policies(self) -> list[PolicyResult]:
        """Return all policy documents (for browsing/index view)."""
        return [
            PolicyResult(
                title=doc["title"],
                category=doc.get("category", "General"),
                content=doc["content"],
                source=doc.get("source", ""),
                relevance_score=1.0,
                tags=doc.get("tags", []),
                jurisdiction=doc.get("jurisdiction", "India"),
                last_updated=doc.get("last_updated", ""),
            )
            for doc in self._kb
        ]
