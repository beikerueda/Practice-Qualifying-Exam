import streamlit as st
import base64
import re
import html
from deep_translator import GoogleTranslator
from pypdf import PdfReader
import os
import json
import requests
from datetime import datetime

st.set_page_config(layout="wide")

# =====================================================
# LOAD LOCAL PDF FILES
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pdf_files = [
    f for f in os.listdir(BASE_DIR)
    if f.lower().endswith(".pdf")
]

if not pdf_files:
    st.error("No PDF files found in the application folder.")
    st.stop()

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "current_question" not in st.session_state:
    st.session_state.current_question = 1

if "exam_started" not in st.session_state:
    st.session_state.exam_started = False

if "exam_finished" not in st.session_state:
    st.session_state.exam_finished = False

if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

if "score" not in st.session_state:
    st.session_state.score = 0

TOTAL_QUESTIONS = 6
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "novaforgeai/gemma2:2b-optimized"
OLLAMA_NUM_PREDICT = 700
OLLAMA_TEMPERATURE = 0.2

if "questions" not in st.session_state:
    st.session_state.questions = []

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

if "last_llm_error" not in st.session_state:
    st.session_state.last_llm_error = ""

if "last_pdf" not in st.session_state:
    st.session_state.last_pdf = None

if "rationales" not in st.session_state:
    st.session_state.rationales = {}

if "awaiting_next" not in st.session_state:
    st.session_state.awaiting_next = False

if "current_rationale" not in st.session_state:
    st.session_state.current_rationale = ""

if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = "Candidate"

if "answer_history" not in st.session_state:
    st.session_state.answer_history = {}

if "preprocess_mode" not in st.session_state:
    st.session_state.preprocess_mode = True

if "processed_snippet" not in st.session_state:
    st.session_state.processed_snippet = ""

# =====================================================
# CACHED PDF HELPERS
# =====================================================
@st.cache_data(show_spinner=False)
def read_pdf_full_text(path):
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)

@st.cache_data(show_spinner=False)
def read_pdf_bytes(path):
    with open(path, "rb") as f:
        return f.read()

# =====================================================
# TRANSLATION
# =====================================================
excluded_words = {
    "the", "a", "an", "and", "or", "to", "of", "in",
    "on", "at", "for", "with", "by", "is", "are",
    "this", "that", "within", "according", "your"
}

@st.cache_data(show_spinner=False)
def translate_word(word):
    try:
        return GoogleTranslator(source='en', target='pt').translate(word)
    except Exception:
        return word

@st.cache_data(show_spinner=False)
def add_hover_translation(text):
    words = re.findall(r"\w+|\W+", text)
    new_text = []

    for word in words:
        clean_word = word.lower()
        escaped_word = html.escape(word)
        if clean_word.isalpha() and clean_word not in excluded_words:
            translated = translate_word(clean_word)
            new_text.append(
                f'<span title="{html.escape(translated, quote=True)}" style="cursor: help;">{escaped_word}</span>'
            )
        else:
            new_text.append(escaped_word)

    return "".join(new_text)

# =====================================================
# EXTRACT TITLES
# =====================================================
def extract_titles_from_pdf(path):
    full_text = read_pdf_full_text(path)

    lines = full_text.split("\n")
    titles = []

    for line in lines:
        line = line.strip()
        if len(line) < 120 and re.match(r"^\d+\.\s+[A-Za-z].*", line):
            titles.append(line)

    return list(dict.fromkeys(titles))

# =====================================================
# EXTRACT TOPIC SNIPPET
# =====================================================
def extract_topic_snippet(path, topic, max_chars=1500):
    full_text = read_pdf_full_text(path)

    lines = [line.strip() for line in full_text.split("\n") if line.strip()]

    if topic == "No structured titles found":
        return full_text[:max_chars]

    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if line == topic:
            start_idx = i + 1
            break

    if start_idx is None:
        return full_text[:max_chars]

    for j in range(start_idx, len(lines)):
        if re.match(r"^\d+\.\s+[A-Za-z].*", lines[j]):
            end_idx = j
            break

    snippet_lines = lines[start_idx:end_idx] if end_idx else lines[start_idx:]
    snippet = "\n".join(snippet_lines).strip()

    if not snippet:
        return full_text[:max_chars]

    return snippet[:max_chars]

def clean_snippet_text(snippet, max_chars=3500):
    lines = [re.sub(r"\s+", " ", line).strip() for line in snippet.splitlines()]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)
    return cleaned[:max_chars]

def build_preprocess_prompt(snippet, topic):
    return (
        "You are a technical content preprocessor for certification exams.\n"
        "Clean the source text, remove noise, and extract high-value key points.\n"
        "Return ONLY valid JSON in this schema:\n"
        "{\n"
        '  "clean_text": "string",\n'
        '  "key_points": ["string", "string"],\n'
        '  "summary": "string"\n'
        "}\n"
        "Rules:\n"
        "- Keep content faithful to the original text.\n"
        "- No invented facts.\n"
        "- Keep output concise and technical.\n"
        f"Topic: {topic}\n"
        "Source text:\n"
        f"{snippet}\n"
    )

# =====================================================
# LLM (OLLAMA) QUESTION GENERATION
# =====================================================
def build_question_prompt(snippet, topic, total_questions=TOTAL_QUESTIONS):
    return (
        "You are an exam question generator.\n"
        "Given the text snippet below, create exactly "
        f"{total_questions} multiple-choice questions.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "question": "string",\n'
        '      "options": {\n'
        '        "A": "string",\n'
        '        "B": "string",\n'
        '        "C": "string",\n'
        '        "D": "string"\n'
        "      },\n"
        '      "answer": "A|B|C|D"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Topic: {topic}\n"
        "Text snippet:\n"
        f"{snippet}\n"
    )

def build_rationale_prompt(snippet, topic, question, options, answer):
    return (
        "You are an exam tutor.\n"
        "Explain why the correct answer is correct, based on the snippet.\n"
        "Return ONLY valid JSON with this schema:\n"
        '{ "rationale": "string" }\n'
        f"Topic: {topic}\n"
        "Text snippet:\n"
        f"{snippet}\n"
        "Question:\n"
        f"{question}\n"
        "Options:\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n"
        f"C. {options['C']}\n"
        f"D. {options['D']}\n"
        f"Correct answer: {answer}\n"
    )

def _extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue

    return None

def _extract_json_array(text):
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return None

def _safe_text(value):
    if value is None:
        return ""
    return str(value).strip()

def _extract_questions_payload(raw_text):
    parsed = _extract_json(raw_text)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("questions"), list):
            return parsed["questions"]
        if isinstance(parsed.get("items"), list):
            return parsed["items"]

    arr = _extract_json_array(raw_text)
    if isinstance(arr, list):
        return arr

    return None

def preprocess_snippet_with_ollama(snippet, topic):
    prompt = build_preprocess_prompt(snippet, topic)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE
        },
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    raw_text = data.get("response", "")
    parsed = _extract_json(raw_text)

    if isinstance(parsed, dict):
        clean_text = clean_snippet_text(_safe_text(parsed.get("clean_text", "")))
        key_points = parsed.get("key_points", [])
        if not isinstance(key_points, list):
            key_points = []
        key_points = [_safe_text(item) for item in key_points if _safe_text(item)]
        summary = _safe_text(parsed.get("summary", ""))

        sections = []
        if clean_text:
            sections.append(clean_text)
        if key_points:
            sections.append("Key points:\n- " + "\n- ".join(key_points[:8]))
        if summary:
            sections.append("Summary:\n" + summary)

        processed = "\n\n".join(sections).strip()
        if processed:
            return processed[:3800], prompt

    fallback = clean_snippet_text(snippet)
    if fallback:
        return fallback, prompt

    raise ValueError("Preprocess mode failed to produce usable content.")

def generate_questions_with_ollama(snippet, topic):
    prompt = build_question_prompt(snippet, topic)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE
        },
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()

    data = response.json()
    raw_text = data.get("response", "")

    questions = _extract_questions_payload(raw_text)
    if questions is None:
        repair_prompt = (
            "Convert the following content into valid JSON with this schema only:\n"
            "{ \"questions\": [ { \"question\": \"string\", \"options\": { \"A\": \"string\", \"B\": \"string\", \"C\": \"string\", \"D\": \"string\" }, \"answer\": \"A|B|C|D\" } ] }\n"
            f"Need exactly {TOTAL_QUESTIONS} questions.\n"
            "Content:\n"
            f"{raw_text}\n"
        )
        repair_payload = {
            "model": OLLAMA_MODEL,
            "prompt": repair_prompt,
            "format": "json",
            "options": {
                "num_predict": OLLAMA_NUM_PREDICT,
                "temperature": 0.1
            },
            "stream": False
        }
        repair_response = requests.post(OLLAMA_URL, json=repair_payload, timeout=120)
        repair_response.raise_for_status()
        repair_data = repair_response.json()
        repaired_raw = repair_data.get("response", "")
        questions = _extract_questions_payload(repaired_raw)

    if questions is None:
        raise ValueError("LLM did not return valid JSON with 'questions'.")

    if not isinstance(questions, list) or len(questions) < TOTAL_QUESTIONS:
        raise ValueError(f"LLM returned fewer than {TOTAL_QUESTIONS} questions.")

    normalized = []
    for q in questions[:TOTAL_QUESTIONS]:
        if not isinstance(q, dict):
            continue

        options = q.get("options", {})
        if not isinstance(options, dict):
            options = {}

        answer = _safe_text(q.get("answer", "")).upper()
        if answer not in {"A", "B", "C", "D"}:
            answer = "A"

        normalized.append({
            "question": _safe_text(q.get("question", "")),
            "options": {
                "A": _safe_text(options.get("A", "")),
                "B": _safe_text(options.get("B", "")),
                "C": _safe_text(options.get("C", "")),
                "D": _safe_text(options.get("D", "")),
            },
            "answer": answer
        })

    return normalized, prompt

def generate_rationale_with_ollama(snippet, topic, question, options, answer):
    prompt = build_rationale_prompt(snippet, topic, question, options, answer)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE
        },
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    raw_text = data.get("response", "")
    parsed = _extract_json(raw_text)

    if parsed and "rationale" in parsed and str(parsed["rationale"]).strip():
        return str(parsed["rationale"]).strip(), prompt

    fallback = raw_text.strip()
    if fallback:
        return fallback, prompt

    raise ValueError("LLM did not return valid JSON with 'rationale'.")

# =====================================================
# LAYOUT
# =====================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    :root {
        --ibc-blue-900: #003A70;
        --ibc-blue-700: #005DAA;
        --ibc-blue-500: #1E88E5;
        --ibc-bg: #f3f6f9;
        --ibc-panel: #ffffff;
        --ibc-muted: #e7edf3;
        --ibc-text: #16324a;
    }

    html, body, [class*="css"]  {
        font-family: "Roboto", sans-serif;
        color: var(--ibc-text);
    }

    .stApp {
        background: linear-gradient(180deg, #f9fbfd 0%, var(--ibc-bg) 100%);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.6rem;
        max-width: 1500px;
    }

    .ibc-header {
        background: linear-gradient(110deg, var(--ibc-blue-900) 0%, var(--ibc-blue-700) 70%, var(--ibc-blue-500) 100%);
        border-radius: 14px;
        color: #ffffff;
        padding: 18px 24px;
        display: flex;
        align-items: center;
        gap: 18px;
        box-shadow: 0 10px 28px rgba(0, 58, 112, 0.18);
        margin-bottom: 16px;
    }

    .ibc-logo {
        width: 86px;
        height: 86px;
        object-fit: contain;
        background: rgba(255, 255, 255, 0.96);
        border-radius: 12px;
        padding: 8px;
    }

    .ibc-title {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .ibc-subtitle {
        margin: 3px 0 0 0;
        font-size: 0.95rem;
        opacity: 0.92;
        font-weight: 500;
    }

    .ibc-panel {
        background: var(--ibc-panel);
        border: 1px solid #d7e1ea;
        border-radius: 12px;
        padding: 14px 14px 10px 14px;
        box-shadow: 0 5px 15px rgba(10, 41, 71, 0.08);
        margin-bottom: 12px;
    }

    .ibc-section {
        font-weight: 700;
        color: var(--ibc-blue-900);
        border-left: 4px solid var(--ibc-blue-500);
        background: #eef5fc;
        padding: 6px 10px;
        border-radius: 4px;
        margin: 4px 0 12px 0;
    }

    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--ibc-blue-700), var(--ibc-blue-500));
    }

    div.stButton > button {
        background: var(--ibc-blue-700);
        color: #ffffff;
        border: 1px solid var(--ibc-blue-900);
        border-radius: 8px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background: var(--ibc-blue-900);
        color: #ffffff;
        border-color: var(--ibc-blue-900);
    }

    div[data-baseweb="select"] > div {
        border-radius: 8px;
        border-color: #b8c9da;
        background: #ffffff;
    }

    .ibc-viewer {
        border: 1px solid #cddceb;
        border-radius: 10px;
        overflow: hidden;
        background: #f2f6fa;
    }

    .ibc-result-meta {
        display: grid;
        grid-template-columns: repeat(2, minmax(220px, 1fr));
        gap: 10px;
    }

    .ibc-meta-item {
        background: #f7fbff;
        border: 1px solid #dbe7f3;
        border-radius: 10px;
        padding: 10px 12px;
    }

    .ibc-meta-label {
        font-size: 0.78rem;
        color: #5a748e;
        margin: 0 0 3px 0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 700;
    }

    .ibc-meta-value {
        margin: 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #113251;
    }

    .ibc-score-wrap {
        display: grid;
        grid-template-columns: 190px 1fr;
        gap: 16px;
        align-items: center;
        margin-top: 8px;
    }

    .ibc-gauge {
        width: 170px;
        height: 170px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
    }

    .ibc-gauge-core {
        width: 124px;
        height: 124px;
        border-radius: 50%;
        background: #ffffff;
        border: 1px solid #dbe7f1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .ibc-score-value {
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--ibc-blue-900);
        line-height: 1.1;
    }

    .ibc-status {
        margin-top: 3px;
        font-size: 0.86rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    .ibc-status.pass {
        color: #0d8a4f;
    }

    .ibc-status.fail {
        color: #b3261e;
    }

    .ibc-breakdown-item {
        margin-bottom: 10px;
    }

    .ibc-breakdown-head {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .ibc-breakdown-track {
        width: 100%;
        height: 10px;
        border-radius: 30px;
        background: #dfeaf5;
        overflow: hidden;
    }

    .ibc-breakdown-fill {
        height: 100%;
        border-radius: 30px;
        background: linear-gradient(90deg, var(--ibc-blue-700), var(--ibc-blue-500));
    }

    .ibc-feedback-box {
        border: 1px solid #dbe7f3;
        background: #f8fbff;
        border-radius: 10px;
        padding: 10px 12px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

logo_path = None
for logo_name in ("LOGO.PNG", "LOGO.png"):
    candidate = os.path.join(BASE_DIR, logo_name)
    if os.path.exists(candidate):
        logo_path = candidate
        break

if logo_path:
    logo_b64 = base64.b64encode(read_pdf_bytes(logo_path)).decode("utf-8")
    logo_html = f'<img class="ibc-logo" src="data:image/png;base64,{logo_b64}" alt="IBC logo">'
else:
    logo_html = ""

st.markdown(
    f"""
    <div class="ibc-header">
        {logo_html}
        <div>
            <p class="ibc-title">IBC Technical Certification Examination Portal</p>
            <p class="ibc-subtitle">Professional Industrial Training and Qualification Assessment</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col_left, col_right = st.columns([1.1, 0.9], gap="large")

# =====================================================
# LEFT COLUMN
# =====================================================
with col_left:
    st.markdown('<div class="ibc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="ibc-section">Exam Workspace</div>', unsafe_allow_html=True)

    selected_pdf = st.selectbox(
        "Qualification Standard",
        pdf_files
    )
    pdf_path = os.path.join(BASE_DIR, selected_pdf)
    st.session_state.candidate_name = st.text_input(
        "Candidate Name",
        value=st.session_state.candidate_name
    )
    st.session_state.preprocess_mode = st.toggle(
        "Preprocess topic before generating questions",
        value=st.session_state.preprocess_mode,
        help="Cleans and summarizes topic text before question generation."
    )

    st.markdown('<div class="ibc-section">Select Technical Topic</div>', unsafe_allow_html=True)

    topics = extract_titles_from_pdf(pdf_path)
    if not topics:
        topics = ["No structured titles found"]

    selected_topic = st.selectbox("Choose a topic:", topics)

    # RESET IF TOPIC CHANGES
    if selected_topic != st.session_state.last_topic or pdf_path != st.session_state.last_pdf:
        st.session_state.current_question = 1
        st.session_state.exam_started = False
        st.session_state.exam_finished = False
        st.session_state.score = 0
        st.session_state.questions = []
        st.session_state.last_prompt = ""
        st.session_state.last_llm_error = ""
        st.session_state.rationales = {}
        st.session_state.awaiting_next = False
        st.session_state.current_rationale = ""
        st.session_state.answer_history = {}
        st.session_state.processed_snippet = ""
        st.session_state.last_topic = selected_topic
        st.session_state.last_pdf = pdf_path

    st.divider()

    # =====================================================
    # START EXAM
    # =====================================================
    if not st.session_state.exam_started:
        if st.button("Start Exam"):
            snippet = extract_topic_snippet(pdf_path, selected_topic)
            snippet_for_exam = snippet
            try:
                if st.session_state.preprocess_mode:
                    try:
                        with st.spinner("Preprocessing topic content..."):
                            snippet_for_exam, _ = preprocess_snippet_with_ollama(
                                snippet=snippet,
                                topic=selected_topic
                            )
                    except Exception as preprocess_error:
                        st.session_state.last_llm_error = f"Preprocess mode fallback: {preprocess_error}"
                        snippet_for_exam = snippet
                st.session_state.processed_snippet = snippet_for_exam

                with st.spinner("Generating questions with local LLM..."):
                    questions, prompt = generate_questions_with_ollama(
                        snippet=snippet_for_exam,
                        topic=selected_topic
                    )
                st.session_state.questions = questions
                st.session_state.last_prompt = prompt
                st.session_state.last_llm_error = ""
            except Exception as e:
                st.session_state.last_llm_error = str(e)
                st.error(f"LLM error: {e}")
                st.stop()

            st.session_state.exam_started = True
            st.session_state.exam_finished = False
            st.session_state.current_question = 1
            st.session_state.score = 0
            st.session_state.awaiting_next = False
            st.session_state.current_rationale = ""
            st.session_state.answer_history = {}
            st.rerun()

    # =====================================================
    # QUESTIONS
    # =====================================================
    if st.session_state.exam_started and not st.session_state.exam_finished:

        st.subheader(f"Question {st.session_state.current_question}")

        if st.session_state.questions:
            q_index = st.session_state.current_question - 1
            current_q = st.session_state.questions[q_index]
            question_text = current_q["question"]
            options_list = [
                f"A. {current_q['options']['A']}",
                f"B. {current_q['options']['B']}",
                f"C. {current_q['options']['C']}",
                f"D. {current_q['options']['D']}",
            ]
            answer_key = current_q["answer"]
            if answer_key not in current_q["options"]:
                answer_key = "A"
            correct_answer = f"{answer_key}. {current_q['options'][answer_key]}"
        else:
            question_text = (
                f"This is question {st.session_state.current_question} "
                f"within {selected_topic}. Select A, B, C or D."
            )
            options_list = [
                "A. Answer 1",
                "B. Answer 2",
                "C. Answer 3",
                "D. Answer 4"
            ]
            correct_answer = "A. Answer 1"

        st.markdown(add_hover_translation(question_text), unsafe_allow_html=True)

        answer = st.radio(
            "Choose an alternative:",
            options_list,
            key=f"answer_{st.session_state.current_question}",
            disabled=st.session_state.awaiting_next
        )

        if st.button(
            "Confirm Answer",
            key=f"confirm_{st.session_state.current_question}",
            disabled=st.session_state.awaiting_next
        ):

            if answer == correct_answer:
                st.session_state.score += 1
                st.session_state.answer_history[st.session_state.current_question] = True
                st.session_state.awaiting_next = False
                st.session_state.current_rationale = ""
                if st.session_state.current_question < TOTAL_QUESTIONS:
                    st.session_state.current_question += 1
                    st.rerun()
                else:
                    st.session_state.exam_finished = True
                    st.rerun()
            else:
                st.session_state.answer_history[st.session_state.current_question] = False
                if st.session_state.questions:
                    rationale_key = f"{selected_topic}:{q_index}"
                    if rationale_key not in st.session_state.rationales:
                        try:
                            snippet = st.session_state.processed_snippet or extract_topic_snippet(pdf_path, selected_topic)
                            with st.spinner("Getting rationale for incorrect answer..."):
                                rationale, prompt = generate_rationale_with_ollama(
                                    snippet=snippet,
                                    topic=selected_topic,
                                    question=current_q["question"],
                                    options=current_q["options"],
                                    answer=current_q["answer"]
                                )
                            st.session_state.rationales[rationale_key] = rationale
                            st.session_state.last_prompt = prompt
                            st.session_state.last_llm_error = ""
                        except Exception as e:
                            st.session_state.last_llm_error = str(e)
                            st.session_state.rationales[rationale_key] = (
                                "Could not generate rationale at this moment."
                            )

                rationale_key = f"{selected_topic}:{q_index}"
                if rationale_key in st.session_state.rationales:
                    st.session_state.current_rationale = st.session_state.rationales[rationale_key]
                else:
                    st.session_state.current_rationale = "No rationale available."
                st.session_state.awaiting_next = True

        if st.session_state.awaiting_next:
            st.info(st.session_state.current_rationale)
            if st.button("Next Question", key=f"next_{st.session_state.current_question}"):
                st.session_state.awaiting_next = False
                st.session_state.current_rationale = ""
                if st.session_state.current_question < TOTAL_QUESTIONS:
                    st.session_state.current_question += 1
                else:
                    st.session_state.exam_finished = True
                st.rerun()

    # =====================================================
    # PROGRESS
    # =====================================================
    st.subheader("Learning Progress")

    progress = st.session_state.current_question / TOTAL_QUESTIONS
    st.progress(min(progress, 1.0))

    st.write(f"Current Question: {min(st.session_state.current_question, TOTAL_QUESTIONS)} of {TOTAL_QUESTIONS}")

    # =====================================================
    # LLM PROMPT (OPTIONAL)
    # =====================================================
    if st.session_state.last_prompt:
        with st.expander("Show LLM Prompt"):
            st.code(st.session_state.last_prompt, language="text")

    if st.session_state.preprocess_mode and st.session_state.processed_snippet:
        with st.expander("Show Processed Topic Content"):
            st.text(st.session_state.processed_snippet)

    if st.session_state.last_llm_error:
        st.warning(st.session_state.last_llm_error)

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
# FEEDBACK SECTION (BASED ON TOPIC)
# =====================================================

st.markdown('<div class="ibc-panel">', unsafe_allow_html=True)
st.markdown('<div class="ibc-section">Performance Feedback</div>', unsafe_allow_html=True)

topic = selected_topic
percentage = (st.session_state.score / TOTAL_QUESTIONS) * 100 if TOTAL_QUESTIONS else 0

if st.session_state.exam_finished:
    cert_name = os.path.splitext(selected_pdf)[0]
    cert_slug = re.sub(r"[^A-Z0-9]", "", cert_name.upper())
    exam_id = f"IBC-{cert_slug[:10] if cert_slug else 'EXAM'}"
    exam_date = datetime.now().strftime("%B %d, %Y")
    candidate_name = st.session_state.candidate_name.strip() or "Candidate"
    pass_mark = 70
    status = "PASS" if percentage >= pass_mark else "FAIL"
    status_class = "pass" if status == "PASS" else "fail"
    gauge_degrees = int(max(0, min(100, percentage)) * 3.6)

    st.markdown(
        f"""
        <div class="ibc-result-meta">
            <div class="ibc-meta-item"><p class="ibc-meta-label">Exam ID</p><p class="ibc-meta-value">{html.escape(exam_id)}</p></div>
            <div class="ibc-meta-item"><p class="ibc-meta-label">Certification</p><p class="ibc-meta-value">{html.escape(cert_name)}</p></div>
            <div class="ibc-meta-item"><p class="ibc-meta-label">Exam Date</p><p class="ibc-meta-value">{html.escape(exam_date)}</p></div>
            <div class="ibc-meta-item"><p class="ibc-meta-label">Candidate Name</p><p class="ibc-meta-value">{html.escape(candidate_name)}</p></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="ibc-score-wrap">
            <div class="ibc-gauge" style="background: conic-gradient(#1E88E5 {gauge_degrees}deg, #dbe8f5 0deg);">
                <div class="ibc-gauge-core">
                    <p class="ibc-score-value">{percentage:.0f}%</p>
                    <div class="ibc-status {status_class}">{status}</div>
                </div>
            </div>
            <div>
                <p class="ibc-meta-label">Final Score</p>
                <p class="ibc-meta-value" style="font-size:1.2rem; margin-bottom:4px;">{st.session_state.score} / {TOTAL_QUESTIONS} Correct Answers</p>
                <p class="ibc-meta-value" style="font-weight:500;">Passing Threshold: {pass_mark}%</p>
                <p class="ibc-meta-value" style="font-weight:500;">Primary Topic: {html.escape(topic)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    domain_labels = [
        "Domain 1 - Fundamentals",
        "Domain 2 - Procedures",
        "Domain 3 - Interpretation",
        "Domain 4 - Applied Practice",
    ]
    chunk = max(1, TOTAL_QUESTIONS // len(domain_labels))
    domain_ranges = []
    start = 1
    for idx, label in enumerate(domain_labels):
        if start > TOTAL_QUESTIONS:
            break
        end = start + chunk - 1
        if idx == len(domain_labels) - 1:
            end = TOTAL_QUESTIONS
        end = min(end, TOTAL_QUESTIONS)
        domain_ranges.append((label, start, end))
        start = end + 1

    bars_html = []
    for label, start_q, end_q in domain_ranges:
        outcomes = [
            st.session_state.answer_history.get(i)
            for i in range(start_q, end_q + 1)
            if st.session_state.answer_history.get(i) is not None
        ]
        domain_score = (
            (sum(1 for val in outcomes if val) / len(outcomes)) * 100
            if outcomes else 0
        )
        bars_html.append(
            f"""
            <div class="ibc-breakdown-item">
                <div class="ibc-breakdown-head">
                    <span>{html.escape(label)}</span>
                    <span>{domain_score:.0f}%</span>
                </div>
                <div class="ibc-breakdown-track">
                    <div class="ibc-breakdown-fill" style="width:{domain_score:.0f}%;"></div>
                </div>
            </div>
            """
        )

    st.markdown('<div class="ibc-section" style="margin-top:14px;">Topic Performance Breakdown</div>', unsafe_allow_html=True)
    st.markdown("".join(bars_html), unsafe_allow_html=True)

    if status == "PASS":
        feedback_title = "Certification Decision: Candidate meets the required competency standard."
        feedback_body = (
            "<ul>"
            "<li>Demonstrates reliable comprehension of the selected standard.</li>"
            "<li>Shows consistent decision quality across assessment domains.</li>"
            "<li>Recommended next step: proceed to advanced module or practical validation.</li>"
            "</ul>"
        )
    else:
        feedback_title = "Certification Decision: Candidate does not yet meet the competency threshold."
        feedback_body = (
            "<ul>"
            "<li>Priority focus: review missed domains and reinforce standard interpretation.</li>"
            "<li>Practice with scenario-based questions before retaking the exam.</li>"
            "<li>Recommended next step: targeted remediation on weak performance areas.</li>"
            "</ul>"
        )

    st.markdown(
        f"""
        <div class="ibc-feedback-box">
            <p class="ibc-meta-value" style="margin-bottom:6px;">{feedback_title}</p>
            {feedback_body}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Complete the exam to view your performance feedback.")

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# RIGHT COLUMN (PDF VIEWER)
# =====================================================
with col_right:
    st.markdown('<div class="ibc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="ibc-section">Reference Document Viewer</div>', unsafe_allow_html=True)

    pdf_bytes = read_pdf_bytes(pdf_path)

    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    st.markdown(
        f"""
        <div class="ibc-viewer">
            <iframe 
                src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" 
                height="820px"
                type="application/pdf">
            </iframe>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

