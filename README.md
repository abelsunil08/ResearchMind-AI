# 🧠 ResearchMind AI
**AI-Powered Academic Research Paper Summarizer**  
*College Assignment • Task 2 – Problem 6*

---

## 📌 Project Overview
**ResearchMind AI** is an intelligent web application built using Python, Streamlit, and Google's official `google-genai` SDK. It extracts text from academic research papers in PDF format and generates structured, beginner-friendly summaries tailored for first-year B.Tech Computer Science & Engineering (AI) students.

### Key Capabilities
1. **PDF Text Extraction**: Uses `pypdf` to extract selectable text page-by-page while gracefully handling empty, corrupted, or scanned-image PDFs.
2. **7-Point Structured Analysis**:
   - 📄 Paper Title & Authors
   - 🎯 Problem Statement (What problem is the research trying to solve?)
   - 🔬 Method Used (Technology, architecture, and algorithms)
   - 📊 Dataset Details (Data sources and benchmarks)
   - 🏆 Main Results & Findings
   - ⚠️ Reported Limitations
   - 💡 Five Key Takeaways (Numbered 1 to 5)
3. **Zero-Hallucination Guardrails**: Strictly grounds answers in the paper text; any missing information explicitly outputs *"Not mentioned in the paper"*.
4. **Interactive Dashboard**: Modern dark navy & electric blue theme with responsive design for desktop and mobile.
5. **Downloadable Report**: Export the complete structured summary as a clean, formatted `.txt` report.

---

## 🛠️ Technology Stack
- **Programming Language**: Python (3.10+)
- **Web UI Framework**: Streamlit
- **AI Engine**: Google Gemini API (`gemini-2.5-flash`) via the official `google-genai` Python SDK
- **PDF Extraction**: `pypdf`
- **Secrets Management**: `python-dotenv`

---

## 🚀 Setup & Execution Guide

### 1. Clone or Extract Project
Open a terminal in the project directory:
```powershell
cd ResearchMind_AI
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Your Gemini API Key
1. Obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).
2. Create a `.env` file in the project root (or copy `.env.example` to `.env`):
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

### 5. Launch the Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Project Directory Structure
```
ResearchMind_AI/
├── app.py              # Main Streamlit web application & AI logic
├── requirements.txt    # Project dependencies
├── .env.example        # Template for environment configuration
├── .gitignore          # Git exclusion rules (ignores .env and .venv)
└── README.md           # Project documentation
```

---

## 🔒 Security Best Practices
- The secret `GEMINI_API_KEY` is loaded strictly via environment variables using `python-dotenv`.
- The `.env` file is excluded from version control and submission ZIPs via `.gitignore`.
- API keys are never hardcoded or printed on the application screen.
