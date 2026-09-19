# ♻️ EcoSort AI — Enterprise Waste Intelligence & Policy Copilot

**EcoSort AI** is an enterprise-grade AI solution engineered to streamline municipal waste segregation, provide automated regulatory compliance guidance, and deliver real-time environmental impact forecasting. Developed to support smart zero-waste communities, educational campuses, and urban local bodies (ULBs), the platform integrates Computer Vision, Retrieval-Augmented Generation (RAG), and Conversational AI into a unified web application.

The project is aligned with the mandates of **Swachh Bharat Mission-Urban 2.0 (SBM-U 2.0)**, the **Solid Waste Management (SWM) Rules 2016**, **Mission LiFE (Lifestyle for Environment)**, and the **United Nations Sustainable Development Goals (SDG 11 & SDG 12)**.

---

## 📌 Project Overview

Rapid urban growth and inadequate source segregation place a severe strain on municipal waste processing infrastructure. EcoSort AI addresses this operational gap by combining automated visual inspection with an authoritative legal knowledge base:

* **Visual Waste Triage:** Immediate identification of item category, recyclability, and appropriate bin color codes.
* **Statutory Compliance:** On-demand interpretation of waste disposal bylaws and bulk generator rules without manual document searches.
* **Data-Driven Impact Projection:** Predictive forecasting of landfill volume saved, community diversion rates, and net greenhouse gas reductions.

---

## 🌟 Core Modules & Capabilities

### 1. 🔍 Multimodal Live Waste Analyzer & Area Audit
* **Single-Item Inference:** Identifies individual waste items via image upload, real-time camera feed, or textual queries. It outputs the exact waste category (Dry, Wet, Domestic Hazardous, E-Waste/Sanitary), appropriate disposal protocols, and decomposition life cycle data.
* **Garbage Area Audit:** Analyzes photos of community dump sites or unsegregated piles. It provides fractional composition estimates and formulates an actionable cleanup and clearance plan.

### 2. 🤖 IBM Bob — Conversational Assistant
* **Interactive Triage:** Powers conversational interactions, allowing users to ask natural questions regarding segregation dilemmas, item handling procedures, and recycling routes.
* **Workflow Coordination:** Interprets user intent to direct queries smoothly between visual classification tools and regulatory resources.

### 3. 📜 RAG-Powered Policy Assistant
* **13 Indexed Policy Documents:** Houses an indexed regulatory repository containing national directives, central ministry guidelines, and statutory solid waste frameworks (including SBM-U 2.0 circulars and SWM Rules 2016).
* **Retrieval-Augmented Generation (RAG):** Extracts precise statutory clauses relevant to the user query and grounds model responses strictly within official regulations, eliminating hallucinated policies.

### 4. 🌍 Predictive Environmental Impact & Carbon Metrics
* **Dynamic Simulation Engine:** Evaluates scenarios for Residential Societies, University Campuses, and Commercial Facilities based on population inputs and active timeframes.
* **Standardized CPHEEO Benchmarks:** Uses standard per-capita urban baseline indicators (0.73 kg/capita/day) to compute:
  * **Total Waste Generated** (kg)
  * **Diversion Rate** (% diverted from landfills)
  * **Net Carbon Offset** ($\text{t }\text{CO}_2\text{e}$)
  * **Tree Absorption Equivalents** (trees/year)
  * **Landfill Space Saved** ($\text{m}^3$)

### 5. 🛡️ Responsible AI & Governance
* Transparent reporting of inference confidence scores, item composition breakdowns, and clear source attribution for every segregation recommendation.

---

## 🛠️ Complete Technology Stack

| Layer / Component | Technology Used | Purpose |
| :--- | :--- | :--- |
| **Frontend Framework** | Streamlit (v1.60.0) | Interactive dashboard and UI rendering |
| **Styling & Theming** | Custom CSS3 Overrides | White-label branding, persistent controls, and theme lock |
| **Conversational Interface** | IBM Bob Framework | Dialog flow management and natural language assistance |
| **Vision & AI Engine** | Google Gemini Multimodal Vision API | Image-based waste classification and dump area audits |
| **Knowledge Engine** | Retrieval-Augmented Generation (RAG) | Vector-indexed search over 13 statutory policy frameworks |
| **Core Language** | Python 3.10+ | Business logic, calculations, and backend pipeline |
| **Program Initiative** | 1M1B & IBM SkillsBuild | Developed under the AI for Sustainability internship framework |

---

## 🏛️ Policy, Standards & Regulatory Alignment

* **MoHUA SBM-Urban 2.0:** Standardized 3-way/4-way waste stream source segregation protocols.
* **Solid Waste Management Rules 2016:** Statutory handling of municipal, organic, and hazardous fractions.
* **Mission LiFE:** Encouraging pro-planet citizen behavior through actionable recycling knowledge.
* **UN Sustainable Development Goals:**
  * **Target 11.6:** Reducing adverse urban environmental impact per capita.
  * **Target 12.5:** Substantially reducing waste generation through prevention, reduction, recycling, and reuse.

---

## 🚀 Step-by-Step Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/EcoSort-AI.git
cd EcoSort-AI
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# .streamlit/secrets.toml
GEMINI_API_KEY = "your_google_gemini_api_key_here"

# Run the Application
streamlit run app.py
Access the application locally at http://localhost:8501.
📁 Repository Structure
Plaintext
├── .streamlit/
│   ├── config.toml           # Theme configuration
│   └── secrets.toml          # API credentials (git-ignored)
├── assets/                   # UI badges, sample test photos, and static assets
├── data/
│   └── policy_docs/          # 13 Indexed statutory policy documents for RAG
├── app.py                    # Primary Streamlit application entry point
├── requirements.txt          # Python dependencies
├── LICENSE                   # Project license
└── README.md                 # Complete documentation
---
##👨‍💻 Developer & Author
Created & Engineered by: Anshu Kumar Gupta

Role: Lead AI & Full-Stack Developer

Project Name: EcoSort AI 

Program Affiliation: Developed under the AI for Sustainability virtual internship initiative conducted by 1M1B in collaboration with IBM SkillsBuild.

## 📄 Copyright & Rights
All Rights Reserved © 2026 Anshu Kumar Gupta.  
Created for academic, internship evaluation, and demonstration purposes.
