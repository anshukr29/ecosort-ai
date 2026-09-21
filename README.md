# 🌱 EcoSort AI — AI-Powered Waste Analyzer & Segregation Copilot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecosort-anshu.streamlit.app)
[![Built with IBM Bob](https://img.shields.io/badge/Developed%20with-IBM%20Bob-052FAD.svg)](https://github.com/anshukr29/ecosort-ai)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**EcoSort AI** is an intelligent waste classification and worker-safety copilot built to solve waste contamination at the disposal point and safeguard municipal sanitation workers from hazardous materials.

Developed as a capstone project for the **AI for Sustainability Virtual Internship** organized by **1M1B**, **IBM SkillsBuild**, and **AICTE**.

---

## 🚀 Live Demo & Repository
* **🌐 Live Web Application:** [ecosort-anshu.streamlit.app](https://ecosort-anshu.streamlit.app)
* **🔗 GitHub Repository:** [github.com/anshukr29/ecosort-ai](https://github.com/anshukr29/ecosort-ai)

---

## 🎯 The Problem
* **Packaging Confusion:** Everyday items (greasy food packets, composite packaging) are often sorted incorrectly, ruining entire batches of recyclable dry waste.
* **Worker Hazards:** Unsegregated household waste often contains broken glass, medical sharps, and caustic chemicals that directly cause preventable injuries to sanitation workers.
* **Landfill Methane Emissions:** Organic and recyclable waste trapped in unmanaged dump yards breaks down into harmful greenhouse gases (GHG).

---

## 💡 Key Features
* 📸 **Multimodal Recyclability Scan:** Real-time visual identification of waste materials, checking recyclability and mapping items to color-coded bins (Blue for Dry/Recyclable, Green for Wet/Organic).
* ⚠️ **Sanitation Worker Safety Alerts:** Automatically detects sharp, chemical, and bio-hazardous waste, providing explicit safe-handling precautions.
* 🌲 **GHG & Tree Impact Simulator:** Computes real-time landfill diversion, avoided carbon emissions ($\text{CO}_2\text{e}$), and converts savings into intuitive **"trees saved"** equivalents based on CPHEEO baselines.
* 📚 **Statutory Waste Policy Assistant:** A dedicated RAG engine providing verified municipal guidelines and waste disposal bylaws grounded in India's **Solid Waste Management (SWM) Rules 2016**.

---

## 🤖 Agentic Development with IBM Bob
This application was engineered using **IBM Bob** for agentic software workflows:
* **Autonomous Workspace Scaffolding:** Structured modular components (`app.py`, `classifier.py`, `rag_engine.py`, `vision_engine.py`).
* **Agentic Execution & In-Workspace Debugging:** Leveraged autonomous terminal execution, dependency configuration, and live error resolution to accelerate the development lifecycle.

---

## 🛠️ Tech Stack & Architecture
* **Frontend / UI:** Streamlit
* **Programming Language:** Python 3.10+
* **Vision & Core Reasoning:** Google Gemini API
* **Agentic Pair Programming:** IBM Bob
* **Standards & Knowledge Base:** Custom Policy RAG & CPHEEO Urban Standards

---

## 📂 Project Structure
```text
ecosort-ai/
├── .bob/                  # IBM Bob agent interaction logs and workspace cache
├── .streamlit/            # Streamlit theme & server configuration
├── core/
│   ├── classifier.py      # Core waste classification engine
│   ├── rag_engine.py      # Policy retrieval augmented generation
│   └── vision_engine.py   # Multimodal image analysis pipeline
├── data/
│   └── waste_catalog.json # Waste categories, materials, and emission benchmarks
├── app.py                 # Main Streamlit web application
├── requirements.txt       # Project dependencies
```

---

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/anshukr29/ecosort-ai.git](https://github.com/anshukr29/ecosort-ai.git)
cd ecosort-ai
```

2. **Create a virtual environment & install dependencies:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

3. **Set up your environment variables:**
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

4. **Launch the application:**
```bash
streamlit run app.py
```
