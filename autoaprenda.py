import streamlit as st
import base64
import re
from deep_translator import GoogleTranslator
from pypdf import PdfReader
import os
import json
import requests

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
# TITLE
# =====================================================
st.title("Practice Qualifying Exam: Part 1 (Open Book)")
st.subheader("Standard Qualification")

selected_pdf = st.selectbox(
    "Choose a qualification standard:",
    pdf_files
)

pdf_path = os.path.join(BASE_DIR, selected_pdf)

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

TOTAL_QUESTIONS = 12
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

if "questions" not in st.session_state:
    st.session_state.questions = []

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

if "last_llm_error" not in st.session_state:
    st.session_state.last_llm_error = ""

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
    except:
        return word

def add_hover_translation(text):
    words = re.findall(r"\w+|\W+", text)
    new_text = []

    for word in words:
        clean_word = word.lower()
        if clean_word.isalpha() and clean_word not in excluded_words:
            translated = translate_word(clean_word)
            new_text.append(
                f'<span title="{translated}" style="cursor: help;">{word}</span>'
            )
        else:
            new_text.append(word)

    return "".join(new_text)

# =====================================================
# EXTRACT TITLES
# =====================================================
def extract_titles_from_pdf(path):
    reader = PdfReader(path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    lines = full_text.split("\n")
    titles = []

    for line in lines:
        line = line.strip()
        if len(line) < 120 and re.match(r"^\d+\.\s+[A-Za-z].*", line):
            titles.append(line)

    return list(dict.fromkeys(titles))

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
        '      "answer": "A|B|C|D",\n'
        '      "rationale": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Topic: {topic}\n"
        "Text snippet:\n"
        f"{snippet}\n"
    )

def _extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return None

def generate_questions_with_ollama(snippet, topic):
    prompt = build_question_prompt(snippet, topic)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    raw_text = data.get("response", "")

    parsed = _extract_json(raw_text)
    if not parsed or "questions" not in parsed:
        raise ValueError("LLM did not return valid JSON with 'questions'.")

    questions = parsed["questions"]
    if not isinstance(questions, list) or len(questions) < TOTAL_QUESTIONS:
        raise ValueError("LLM returned fewer than 12 questions.")

    normalized = []
    for q in questions[:TOTAL_QUESTIONS]:
        options = q.get("options", {})
        normalized.append({
            "question": q.get("question", "").strip(),
            "options": {
                "A": options.get("A", "").strip(),
                "B": options.get("B", "").strip(),
                "C": options.get("C", "").strip(),
                "D": options.get("D", "").strip(),
            },
            "answer": q.get("answer", "").strip().upper(),
            "rationale": q.get("rationale", "").strip()
        })

    return normalized, prompt

# =====================================================
# LAYOUT
# =====================================================
col_left, col_right = st.columns([1, 1])

# =====================================================
# LEFT COLUMN
# =====================================================
with col_left:

    st.subheader("Select a Topic")

    topics = extract_titles_from_pdf(pdf_path)
    if not topics:
        topics = ["No structured titles found"]

    selected_topic = st.selectbox("Choose a topic:", topics)

    # RESET IF TOPIC CHANGES
    if selected_topic != st.session_state.last_topic:
        st.session_state.current_question = 1
        st.session_state.exam_started = False
        st.session_state.exam_finished = False
        st.session_state.score = 0
        st.session_state.last_topic = selected_topic

    st.divider()

    # =====================================================
    # LLM INPUT
    # =====================================================
    st.subheader("Generate Questions from Text")

    snippet = st.text_area(
        "Paste a text snippet (English) to generate 12 questions:",
        height=200
    )

    if st.button("Generate Questions"):
        if not snippet.strip():
            st.warning("Please paste a text snippet before generating.")
        else:
            try:
                questions, prompt = generate_questions_with_ollama(
                    snippet=snippet,
                    topic=selected_topic
                )
                st.session_state.questions = questions
                st.session_state.last_prompt = prompt
                st.session_state.last_llm_error = ""
                st.success("Questions generated successfully.")
            except Exception as e:
                st.session_state.last_llm_error = str(e)
                st.error(f"LLM error: {e}")

    if st.session_state.last_prompt:
        with st.expander("Show LLM Prompt"):
            st.code(st.session_state.last_prompt, language="text")

    if st.session_state.last_llm_error:
        st.warning(st.session_state.last_llm_error)

    # =====================================================
    # START EXAM
    # =====================================================
    if not st.session_state.exam_started:
        if st.button("Start Exam"):
            st.session_state.exam_started = True
            st.session_state.exam_finished = False
            st.session_state.current_question = 1
            st.session_state.score = 0
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
            correct_answer = f"{current_q['answer']}. {current_q['options'][current_q['answer']]}"
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
            key=f"answer_{st.session_state.current_question}"
        )

        if st.button("Confirm Answer", key=f"confirm_{st.session_state.current_question}"):

            if answer == correct_answer:
                st.session_state.score += 1

            if st.session_state.current_question < TOTAL_QUESTIONS:
                st.session_state.current_question += 1
                st.rerun()
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
# FEEDBACK SECTION (BASED ON TOPIC)
# =====================================================

st.subheader("Performance Feedback")

topic = selected_topic
percentage = (st.session_state.score / TOTAL_QUESTIONS) * 100 if TOTAL_QUESTIONS else 0

if percentage >= 80:
    st.success(f"You demonstrate strong understanding in '{topic}'.")
    st.write("Your strengths are conceptual clarity and confident decision-making.")
    st.write(f"We recommend exploring advanced aspects of {topic}.")

elif percentage >= 50:
    st.warning(f"You show moderate understanding in '{topic}'.")
    st.write("Your strengths are foundational knowledge and recognition of key concepts.")
    st.write(f"We recommend deepening your study of {topic}.")

else:
    st.error(f"You need improvement in '{topic}'.")
    st.write("You show initial familiarity with the material.")
    st.write(f"We strongly recommend revisiting the section '{topic}' before retaking the exam.")

# =====================================================
# RIGHT COLUMN (PDF VIEWER)
# =====================================================
with col_right:

    st.subheader("Document Viewer")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    st.markdown(
        f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="800px"
            type="application/pdf">
        </iframe>
        """,
        unsafe_allow_html=True
    )
