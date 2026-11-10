"""
ResearchMind AI - Premium AI-Powered Research Paper Summarizer
Task 2 – Problem 6

Design & Features:
1. Premium dark navy & electric blue research dashboard layout.
2. Google Fonts typography (Plus Jakarta Sans & Inter) with clean hierarchy.
3. Interactive 3-step pipeline progress indicator (Upload -> Extract -> Synthesize).
4. Modernized drag-and-drop PDF upload component with glowing border accents.
5. Robust PDF text extraction via pypdf (handles 0-byte, corrupt, scanned PDFs).
6. Google Gemini AI analysis using official google-genai SDK.
7. Prominently highlighted 'Five Key Takeaways' spotlight card.
8. Full 7-section academic breakdown tailored for first-year B.Tech CSE AI students:
   - 1. Paper Title and Authors
   - 2. Problem – What problem is the research trying to solve?
   - 3. Method Used – What method or technology did the researchers use?
   - 4. Dataset – What data was used?
   - 5. Results – What were the main findings?
   - 6. Limitations – What are the limitations mentioned in the paper?
   - 7. Five Key Takeaways – Exactly five important points.
9. One-click markdown summary download for student study notes.
10. Fully responsive layout optimized for mobile screens and laptops.
"""

import io
import os
import re
from dotenv import load_dotenv
import streamlit as st
import pypdf
from pypdf.errors import PdfReadError
from google import genai
from google.genai.errors import APIError

# -----------------------------------------------------------------------------
# 1. Configuration & Secrets Loading
# -----------------------------------------------------------------------------
load_dotenv()

# Single configuration variable for the Gemini Flash model
# gemini-2.5-flash-lite: confirmed working model for this API key tier
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Read API key safely from environment variable (Never hardcoded!)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"].strip()
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# -----------------------------------------------------------------------------
# 2. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ResearchMind AI • Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 3. Custom CSS - Premium Dark Navy & Electric Blue Dashboard Theme
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8f0;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1527 0%, #080c14 60%, #05080e 100%);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 4rem;
        max-width: 1140px;
    }

    /* Top Dashboard Hero Banner */
    .dashboard-hero {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(23, 37, 84, 0.75) 50%, rgba(30, 27, 75, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 35px -5px rgba(2, 6, 23, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(16px);
    }

    .dashboard-hero::before {
        content: '';
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.25) 0%, rgba(99, 102, 241, 0) 70%);
        pointer-events: none;
    }

    .hero-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }

    .brand-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(30, 58, 138, 0.45);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #7dd3fc;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #38bdf8;
        box-shadow: 0 0 10px #38bdf8;
        animation: pulseAnimation 2s infinite ease-in-out;
    }

    @keyframes pulseAnimation {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    .system-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 8px;
    }

    .dashboard-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }

    .dashboard-title span {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .dashboard-welcome {
        font-size: 1.02rem;
        color: #cbd5e1;
        margin-top: 0.85rem;
        margin-bottom: 0;
        line-height: 1.6;
        max-width: 880px;
    }

    /* Pipeline Status Tracker */
    .pipeline-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.9rem;
        margin-bottom: 1.8rem;
    }

    .pipeline-step {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        transition: all 0.25s ease;
    }

    .pipeline-step.step-active {
        border-color: rgba(56, 189, 248, 0.6);
        background: rgba(23, 37, 84, 0.45);
        box-shadow: 0 4px 15px -3px rgba(56, 189, 248, 0.25);
    }

    .pipeline-step.step-done {
        border-color: rgba(16, 185, 129, 0.4);
        background: rgba(6, 78, 59, 0.25);
    }

    .step-number {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
    }

    .step-active .step-number {
        background: #0284c7;
        color: #ffffff;
    }

    .step-done .step-number {
        background: #10b981;
        color: #ffffff;
    }

    .step-info {
        display: flex;
        flex-direction: column;
    }

    .step-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    .step-desc {
        font-size: 0.74rem;
        color: #94a3b8;
    }

    /* Uploader Section Card */
    .upload-card-wrapper {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px -4px rgba(2, 6, 23, 0.5);
    }

    /* Custom Streamlit File Uploader Override */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(56, 189, 248, 0.35) !important;
        border-radius: 16px !important;
        background: rgba(13, 21, 39, 0.5) !important;
        padding: 1rem !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #38bdf8 !important;
        background: rgba(30, 58, 138, 0.15) !important;
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.15) !important;
    }

    /* File Metadata Chip Box */
    .file-details-card {
        background: rgba(23, 37, 84, 0.35);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 14px;
        padding: 0.95rem 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1.3rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.75rem;
    }

    .file-details-left {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-size: 0.92rem;
    }

    .filename-tag {
        font-family: monospace;
        font-size: 0.88rem;
        background: rgba(15, 23, 42, 0.8);
        color: #38bdf8;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* Primary Action Button (Modern Glow) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.6rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.45) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.65) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }

    /* Metric KPI Cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 15px rgba(2, 6, 23, 0.4);
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }

    /* Prominent Five Key Takeaways Spotlight Box */
    .takeaways-spotlight-card {
        background: linear-gradient(135deg, rgba(23, 37, 84, 0.7) 0%, rgba(30, 27, 75, 0.75) 100%);
        border: 1px solid rgba(129, 140, 248, 0.4);
        border-radius: 18px;
        padding: 1.6rem 2rem;
        margin-top: 1.6rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -5px rgba(99, 102, 241, 0.25);
        position: relative;
    }

    .spotlight-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 1.1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        padding-bottom: 0.8rem;
    }

    .spotlight-icon {
        font-size: 1.6rem;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(129, 140, 248, 0.4);
        padding: 0.4rem;
        border-radius: 12px;
        display: inline-flex;
    }

    .spotlight-title {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
    }

    .spotlight-subtitle {
        margin: 0;
        font-size: 0.82rem;
        color: #a5b4fc;
    }

    /* Comprehensive Analysis Container */
    .analysis-container {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 18px;
        padding: 2.2rem 2.4rem;
        margin-top: 1.4rem;
        box-shadow: 0 10px 35px -5px rgba(2, 6, 23, 0.6);
        line-height: 1.7;
    }

    .analysis-container h2, .analysis-container h3 {
        color: #f8fafc;
        border-left: 4px solid #38bdf8;
        padding-left: 0.75rem;
        margin-top: 1.8rem;
        margin-bottom: 0.9rem;
        font-weight: 700;
    }

    .analysis-container strong {
        color: #38bdf8;
    }

    /* Expander Styling */
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 14px !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: rgba(30, 41, 59, 0.85) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button:hover {
        background: rgba(30, 58, 138, 0.4) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.2) !important;
    }

    /* Mobile Responsive Adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        .dashboard-hero {
            padding: 1.5rem 1.2rem;
            border-radius: 16px;
        }
        .dashboard-title {
            font-size: 1.65rem;
        }
        .dashboard-welcome {
            font-size: 0.9rem;
        }
        .pipeline-container {
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }
        .analysis-container {
            padding: 1.3rem;
            border-radius: 14px;
        }
        .takeaways-spotlight-card {
            padding: 1.2rem 1.1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 4. Helper: API Key Validation (Without Displaying Key)
# -----------------------------------------------------------------------------
def is_api_key_configured(key: str) -> bool:
    """Checks whether a non-empty, non-placeholder API key is set."""

    if not key or not key.strip():
        return False

    clean_key = key.strip().lower()

    placeholder_indicators = (
        "your_gemini_api_key_here",
        "your_actual_key_here",
        "your_api_key_here",
        "<your_gemini_api_key_here>",
        "placeholder",
    )

    if clean_key in placeholder_indicators:
        return False

    return True
# -----------------------------------------------------------------------------
def extract_takeaways_spotlight(summary_text: str) -> str:
    """Extracts the 5 Key Takeaways section for prominent display."""
    target_markers = [
        "### 7. 💡 Five Key Takeaways",
        "## 7. 💡 Five Key Takeaways",
        "Five Key Takeaways",
        "7. 💡 Five Key Takeaways",
    ]
    for marker in target_markers:
        if marker in summary_text:
            parts = summary_text.split(marker, 1)
            if len(parts) > 1:
                content = parts[1].strip()
                # Clean leading descriptions if present
                lines = content.split("\n")
                filtered = [
                    l for l in lines 
                    if not (l.strip().startswith("–") or l.strip().startswith("- Exactly"))
                ]
                return "\n".join(filtered).strip()
    return ""

# -----------------------------------------------------------------------------
# 6. Helper: Format Summary for Downloadable Plain Text (.txt) Report
# -----------------------------------------------------------------------------
def format_analysis_as_txt(filename: str, summary_markdown: str) -> str:
    """
    Formats the AI-generated analysis into a clean, human-readable plain text (.txt) report.
    Includes document metadata, all analysis sections, and five key takeaways with neat formatting.
    """
    clean_text = summary_markdown
    # Remove markdown header tokens (###, ##, #)
    clean_text = re.sub(r'#{1,6}\s*', '', clean_text)
    # Remove markdown bold/italic asterisks while preserving text
    clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)
    clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)
    
    divider = "=" * 80
    
    report_lines = [
        divider,
        "                     RESEARCHMIND AI - RESEARCH PAPER SUMMARY",
        divider,
        f" Document Name : {filename}",
        f" Engine        : Google Gemini AI ({GEMINI_MODEL})",
        " Target Level  : First-Year B.Tech CSE AI",
        " Grounding     : Strictly based on paper content (No invented facts)",
        divider,
        "",
        clean_text.strip(),
        "",
        divider,
        "                    END OF SUMMARY • GENERATED BY RESEARCHMIND AI",
        divider,
    ]
    
    return "\n".join(report_lines)

# -----------------------------------------------------------------------------
# 6. Sidebar: Status & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 ResearchMind AI")
    st.caption("College Edition • Task 2 – Problem 6")
    st.markdown("---")
    
    st.markdown(
        """
        **AI Research Assistant**  
        Designed specifically for academic study and first-year B.Tech CSE AI students.
        
        **7-Point Structured Analysis:**
        * 1. 📄 Paper Title & Authors
        * 2. 🎯 Problem Statement
        * 3. 🔬 Method & Tech Used
        * 4. 📊 Dataset Details
        * 5. 🏆 Results & Findings
        * 6. ⚠️ Limitations
        * 7. 💡 Five Key Takeaways
        """
    )

    st.markdown("---")
    st.markdown("### ⚙️ Engine Status")
    st.markdown(f"**Model:** `{GEMINI_MODEL}`")

    # Display API status safely (Never printing the key itself)
    if is_api_key_configured(GEMINI_API_KEY):
        st.success("Gemini API: Key loaded ✅")
        st.caption("Key detected from `.env`. Analysis will confirm connectivity.")
    else:
        st.error("Gemini API: Key not configured ❌")
        st.caption("Add `GEMINI_API_KEY=your_key` (no spaces around `=`) into your `.env` file.")

    st.markdown("---")
    st.info("💡 **Pro-Tip:** Upload papers with digital/selectable text for best extraction accuracy.")

# -----------------------------------------------------------------------------
# 7. Hero Header: Branding & Welcome Message
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="dashboard-hero">
        <div class="hero-top-row">
            <div class="brand-pill">
                <span class="pulse-dot"></span>
                RESEARCHMIND AI • ACADEMIC DASHBOARD
            </div>
            <div class="system-chip">
                ⚡ Powered by Google Gemini AI
            </div>
        </div>
        <h1 class="dashboard-title">
            AI-Powered <span>Research Paper</span> Summarizer
        </h1>
        <p class="dashboard-welcome">
            👋 <strong>Welcome, Researcher!</strong> Transform dense academic literature into crystal-clear, structured insights in seconds. 
            Upload any research paper in PDF format below to inspect document metrics and generate a 7-section academic breakdown tailored for computer science and AI students.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 8. Function: Extract PDF Text with pypdf
# -----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_file) -> dict:
    """
    Extracts selectable text from an uploaded PDF file using pypdf.
    Robustly handles 0-byte, corrupt, empty, and scanned PDFs.
    """
    try:
        pdf_file.seek(0)
        file_bytes = pdf_file.read()

        if len(file_bytes) == 0:
            return {
                "status": "error",
                "message": "The uploaded PDF file is empty (0 bytes). Please upload a valid document.",
                "text": "",
                "page_count": 0,
            }

        pdf_stream = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_stream)

        page_count = len(reader.pages)
        if page_count == 0:
            return {
                "status": "error",
                "message": "The uploaded PDF contains 0 pages.",
                "text": "",
                "page_count": 0,
            }

        extracted_pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text.strip())

        full_text = "\n\n".join(extracted_pages).strip()

        if not full_text:
            return {
                "status": "warning",
                "message": (
                    "No selectable text could be extracted from this PDF. "
                    "The document may consist of scanned images or protected layers. "
                    "Please upload a digital PDF with selectable text."
                ),
                "text": "",
                "page_count": page_count,
            }

        words = full_text.split()
        return {
            "status": "success",
            "message": f"Successfully extracted text from {page_count} page(s).",
            "text": full_text,
            "page_count": page_count,
            "word_count": len(words),
            "char_count": len(full_text),
        }

    except PdfReadError as pe:
        return {
            "status": "error",
            "message": f"Invalid or corrupted PDF file. The document structure could not be parsed: {pe}",
            "text": "",
            "page_count": 0,
        }
    except Exception as ex:
        return {
            "status": "error",
            "message": f"An unexpected error occurred while reading the PDF: {ex}",
            "text": "",
            "page_count": 0,
        }

# -----------------------------------------------------------------------------
# 9. Function: Gemini AI Analysis via google-genai SDK
# -----------------------------------------------------------------------------
def analyze_paper_with_gemini(paper_text: str, model_name: str, api_key: str) -> dict:
    """
    Sends extracted paper text to Gemini AI with 7 structured sections.
    Enforces strict truthfulness ('Not mentioned in the paper' fallback).
    """
    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""You are ResearchMind AI, an expert academic mentor and AI research assistant.
Your task is to analyze the research paper provided below and produce an accessible, high-clarity explanation tailored specifically for a first-year B.Tech Computer Science (CSE) AI student.

CRITICAL INSTRUCTIONS:
1. Base your entire analysis ONLY and STRICTLY on the research paper text provided below.
2. Do NOT invent, assume, extrapolate, or hallucinate any facts, authors, institutions, datasets, or results.
3. If any requested information (such as authors, dataset, limitations, or specific findings) is NOT explicitly mentioned or available in the paper, you MUST write exactly: "Not mentioned in the paper".
4. Language & Tone: Use simple, plain English with intuitive explanations suitable for a beginner B.Tech CSE AI student. Explain technical terms simply and clearly.
5. Provide EXACTLY the 7 sections specified below, using the exact numbered headings and subheadings.

---
### REQUIRED OUTPUT SECTIONS:

### 1. 📄 Paper Title and Authors
- **Title:** [Exact paper title, or "Not mentioned in the paper"]
- **Authors & Affiliations:** [List all author names and their institutions found in the text, or "Not mentioned in the paper"]

### 2. 🎯 Problem – What problem is the research trying to solve?
[Explain the core problem, why it is an important challenge, and the authors' main goal in clear, simple language.]

### 3. 🔬 Method Used – What method or technology did the researchers use?
[Explain the technology, algorithms, models, architecture, or techniques the authors created or implemented. Break it down step-by-step so a first-year student can easily grasp it.]

### 4. 📊 Dataset – What data was used?
[Describe what data, benchmarks, or samples were used for training and testing (names, size, source), or write "Not mentioned in the paper".]

### 5. 🏆 Results – What were the main findings?
[Explain what the researchers achieved, including key performance numbers or qualitative outcomes, or write "Not mentioned in the paper".]

### 6. ⚠️ Limitations – What are the limitations mentioned in the paper?
[State any weaknesses, constraints, hardware limits, or edge cases the authors reported, or write "Not mentioned in the paper".]

### 7. 💡 Five Key Takeaways – Exactly five important points
*Explain exactly 5 key learnings a first-year AI student should remember:*
1. **Takeaway 1:** [First key takeaway]
2. **Takeaway 2:** [Second key takeaway]
3. **Takeaway 3:** [Third key takeaway]
4. **Takeaway 4:** [Fourth key takeaway]
5. **Takeaway 5:** [Fifth key takeaway]

---
RESEARCH PAPER TEXT:
{paper_text}
"""

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        if not response.text:
            return {
                "status": "error",
                "message": (
                    "The AI model returned an empty response. "
                    "The document may have triggered content safety filters or contained unsupported tokens."
                ),
                "summary": "",
            }

        return {
            "status": "success",
            "message": "AI analysis generated successfully!",
            "summary": response.text,
        }

    except APIError as api_err:
        err_str = str(api_err)
        code = getattr(api_err, "code", None)

        if code == 429 or "RESOURCE_EXHAUSTED" in err_str:
            msg = (
                "⚠️ **API Quota Exceeded (HTTP 429)**: The rate limit or free tier quota "
                "for your Gemini API key has been reached. Please wait a moment before trying again, "
                "or review your quota in Google AI Studio."
            )
        elif code == 404 or "NOT_FOUND" in err_str:
            msg = (
                f"⚠️ **Model Unavailable (HTTP 404)**: The model `{model_name}` is not available "
                "for your Google AI Studio account. You can adjust the `GEMINI_MODEL` variable in `app.py` "
                "to an available Flash model (e.g., `gemini-2.0-flash` or `gemini-1.5-flash`)."
            )
        elif code in (400, 403) or "API_KEY_INVALID" in err_str:
            msg = (
                "❌ **Authentication Failed (Invalid API Key)**: The Gemini API key in your `.env` "
                "file is invalid or not authorized. Please verify your key at https://aistudio.google.com/."
            )
        else:
            msg = f"❌ **Gemini API Error (HTTP {code})**: {api_err.message or err_str}"

        return {
            "status": "error",
            "message": msg,
            "summary": "",
        }
    except Exception as ex:
        return {
            "status": "error",
            "message": f"❌ **Unexpected Error during AI analysis**: {str(ex)}",
            "summary": "",
        }

# -----------------------------------------------------------------------------
# 10. Session State Initialization
# -----------------------------------------------------------------------------
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# -----------------------------------------------------------------------------
# 11. Pipeline Progress Tracker Bar
# -----------------------------------------------------------------------------
has_file = st.session_state.last_uploaded_name is not None
has_extracted = (
    st.session_state.extracted_data is not None 
    and st.session_state.extracted_data.get("status") == "success"
)
has_analysis = (
    st.session_state.analysis_result is not None 
    and st.session_state.analysis_result.get("status") == "success"
)

step1_cls = "step-done" if has_file else "step-active"
step2_cls = "step-done" if has_extracted else ("step-active" if has_file else "")
step3_cls = "step-done" if has_analysis else ("step-active" if has_extracted else "")

st.markdown(
    f"""
    <div class="pipeline-container">
        <div class="pipeline-step {step1_cls}">
            <div class="step-number">{'✓' if has_file else '1'}</div>
            <div class="step-info">
                <span class="step-title">1. PDF Document</span>
                <span class="step-desc">{'Document Loaded' if has_file else 'Upload PDF Below'}</span>
            </div>
        </div>
        <div class="pipeline-step {step2_cls}">
            <div class="step-number">{'✓' if has_extracted else '2'}</div>
            <div class="step-info">
                <span class="step-title">2. Text Extraction</span>
                <span class="step-desc">{'Text Extracted' if has_extracted else ('Ready to Extract' if has_file else 'Waiting for File')}</span>
            </div>
        </div>
        <div class="pipeline-step {step3_cls}">
            <div class="step-number">{'✓' if has_analysis else '3'}</div>
            <div class="step-info">
                <span class="step-title">3. AI Synthesis</span>
                <span class="step-desc">{'Synthesis Complete' if has_analysis else ('Ready to Analyze' if has_extracted else 'Pending Extraction')}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 12. PDF Upload Section
# -----------------------------------------------------------------------------
st.markdown('<div class="upload-card-wrapper">', unsafe_allow_html=True)
st.markdown("### 📤 Upload Research Paper (PDF)")
st.caption("Drag and drop your academic paper here. Selectable digital PDF format is recommended.")

uploaded_file = st.file_uploader(
    label="Upload PDF file",
    type=["pdf"],
    help="Supports academic research papers in PDF format with selectable text",
    label_visibility="collapsed",
)

# Clear session state when a new file is uploaded
if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_name:
    st.session_state.extracted_data = None
    st.session_state.analysis_result = None
    st.session_state.last_uploaded_name = uploaded_file.name

# -----------------------------------------------------------------------------
# 13. File Details & Trigger Action
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    file_size_kb = uploaded_file.size / 1024
    size_str = f"{file_size_kb:.1f} KB" if file_size_kb < 1024 else f"{(file_size_kb / 1024):.2f} MB"

    st.markdown(
        f"""
        <div class="file-details-card">
            <div class="file-details-left">
                <span>📄</span>
                <strong>Document:</strong>
                <span class="filename-tag">{uploaded_file.name}</span>
            </div>
            <div>
                <strong>Size:</strong> <span style="color: #38bdf8;">{size_str}</span> &nbsp;|&nbsp;
                <span style="color: #10b981; font-weight: 700;">● Ready for Processing</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn, _ = st.columns([1, 1])
    with col_btn:
        analyze_clicked = st.button(
            "🔬 Analyze Research Paper",
            type="primary",
            use_container_width=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)  # Close upload-card-wrapper

    # Trigger Extraction & AI Analysis
    if analyze_clicked:
        # Step 1: Text Extraction with pypdf
        with st.spinner("Extracting selectable text from PDF..."):
            extracted = extract_text_from_pdf(uploaded_file)
            st.session_state.extracted_data = extracted

        # Step 2: Validate extraction result
        if extracted["status"] == "success":
            # Check if API Key is configured
            if not is_api_key_configured(GEMINI_API_KEY):
                st.session_state.analysis_result = {
                    "status": "error",
                    "message": (
                        "⚠️ **Gemini API Key Missing**: Please set a valid `GEMINI_API_KEY` in your `.env` "
                        "file to enable AI analysis. Get your free key from [Google AI Studio](https://aistudio.google.com/)."
                    ),
                    "summary": "",
                }
            else:
                # Step 3: Send to Google Gemini AI
                with st.spinner(f"Analyzing paper with Google Gemini AI ({GEMINI_MODEL})..."):
                    ai_result = analyze_paper_with_gemini(
                        paper_text=extracted["text"],
                        model_name=GEMINI_MODEL,
                        api_key=GEMINI_API_KEY,
                    )
                    st.session_state.analysis_result = ai_result
        else:
            st.session_state.analysis_result = None

    # -------------------------------------------------------------------------
    # 14. Display Extraction Details & Metrics
    # -------------------------------------------------------------------------
    if st.session_state.extracted_data is not None:
        ext_data = st.session_state.extracted_data

        if ext_data["status"] == "error":
            st.error(f"❌ {ext_data['message']}")

        elif ext_data["status"] == "warning":
            st.warning(f"⚠️ {ext_data['message']}")

        elif ext_data["status"] == "success":
            # KPI Metrics Grid
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📑 Document Pages", ext_data["page_count"])
            with col2:
                st.metric("📝 Extracted Words", f"{ext_data['word_count']:,}")
            with col3:
                st.metric("🔤 Total Characters", f"{ext_data['char_count']:,}")

            # Safe Text Preview Expander
            with st.expander("📄 View Extracted Text Preview", expanded=False):
                preview_text = ext_data["text"][:2500]
                if len(ext_data["text"]) > 2500:
                    preview_text += "\n\n... [Remaining text processed by Gemini AI]"
                st.text_area(
                    label="Extracted Text",
                    value=preview_text,
                    height=200,
                    disabled=True,
                    label_visibility="collapsed",
                )

    # -------------------------------------------------------------------------
    # 15. Display Gemini AI Analysis & Prominent Spotlight
    # -------------------------------------------------------------------------
    if st.session_state.analysis_result is not None:
        ai_data = st.session_state.analysis_result

        if ai_data["status"] == "error":
            st.error(ai_data["message"])

        elif ai_data["status"] == "success":
            st.success("✨ Research Paper Analysis Generated Successfully!")

            # Spotlight: Prominently featured Five Key Takeaways
            takeaways_text = extract_takeaways_spotlight(ai_data["summary"])
            if takeaways_text:
                st.markdown(
                    f"""
                    <div class="takeaways-spotlight-card">
                        <div class="spotlight-header">
                            <span class="spotlight-icon">💡</span>
                            <div>
                                <h3 class="spotlight-title">Five Key Takeaways at a Glance</h3>
                                <p class="spotlight-subtitle">The essential concepts every AI & CS student should remember from this paper</p>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(takeaways_text)

            # Prepare clean plain text (.txt) report for download
            txt_report_content = format_analysis_as_txt(
                filename=uploaded_file.name,
                summary_markdown=ai_data["summary"],
            )
            base_doc_name = os.path.splitext(uploaded_file.name)[0]
            download_filename = f"{base_doc_name}_Summary.txt"

            # Action bar: Download Summary button (Shown ONLY when analysis is available)
            col_head, col_dl = st.columns([2.5, 1.5])
            with col_head:
                st.markdown("### 📋 Complete 7-Section Academic Analysis")
            with col_dl:
                st.download_button(
                    label="📥 Download Summary (.txt)",
                    data=txt_report_content,
                    file_name=download_filename,
                    mime="text/plain",
                    use_container_width=True,
                    help="Download the complete structured summary, including paper title, all analysis sections, and five key takeaways as a readable TXT file.",
                )

            # Full Comprehensive 7-Section Breakdown
            st.markdown(
                f"""
                <div class="analysis-container">
                    <p style="color: #64748b; font-size: 0.88rem; margin-top: 0; margin-bottom: 1.2rem; border-bottom: 1px solid rgba(148, 163, 184, 0.15); padding-bottom: 0.6rem;">
                        <strong>Target Audience:</strong> First-Year B.Tech CSE AI &nbsp;|&nbsp; 
                        <strong>Engine:</strong> {GEMINI_MODEL} &nbsp;|&nbsp; 
                        <strong>Grounding:</strong> Strict Paper Source Only
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(ai_data["summary"])

else:
    st.markdown('</div>', unsafe_allow_html=True)  # Close upload-card-wrapper
    st.session_state.extracted_data = None
    st.session_state.analysis_result = None
    st.session_state.last_uploaded_name = None
    st.info("👆 Please upload a research paper in PDF format above to begin analysis.")
