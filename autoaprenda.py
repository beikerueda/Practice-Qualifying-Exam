import streamlit as st
import base64
import re
from deep_translator import GoogleTranslator
from pypdf import PdfReader
import os

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
    
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None
TOTAL_QUESTIONS = 12

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

        question_text = (
            f"This is question {st.session_state.current_question} "
            f"within {selected_topic}. Select A, B, C or D."
        )

        st.markdown(add_hover_translation(question_text), unsafe_allow_html=True)

        answer = st.radio(
            "Choose an alternative:",
            [
                "A. Answer 1",
                "B. Answer 2",
                "C. Answer 3",
                "D. Answer 4"
            ],
            key=f"answer_{st.session_state.current_question}"
        )

        if st.button("Confirm Answer", key=f"confirm_{st.session_state.current_question}"):

            correct_answer = "A. Answer 1"

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

    st.divider()

# =====================================================
# SCORE SECTION
# =====================================================
st.subheader("Your Final Score")

percentage = (st.session_state.score / TOTAL_QUESTIONS) * 100

# Badge grande colorido
if percentage >= 80:
    badge_color = "#28a745"
elif percentage >= 50:
    badge_color = "#ffc107"
else:
    badge_color = "#dc3545"

st.markdown(
    f"""
    <div style="
        background-color:{badge_color};
        padding:30px;
        border-radius:20px;
        text-align:center;
        color:white;
        font-size:48px;
        font-weight:bold;
    ">
        {percentage:.1f}%
    </div>
    """,
    unsafe_allow_html=True
)

st.write(f"Score: {st.session_state.score} / {TOTAL_QUESTIONS}")

# =====================================================
# PERFORMANCE CHART
# =====================================================

import pandas as pd

correct = st.session_state.score
incorrect = TOTAL_QUESTIONS - correct

data = pd.DataFrame({
    "Result": ["Correct", "Incorrect"],
    "Count": [correct, incorrect]
})

st.subheader("Performance Chart")
st.bar_chart(data.set_index("Result"))

# =====================================================
# FEEDBACK SECTION (BASED ON TOPIC)
# =====================================================

st.subheader("Performance Feedback")

topic = st.session_state.selected_topic

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

with open(selected_pdf, "rb") as f:
    pdf_bytes = f.read()

    st.subheader("Document Viewer")
    st.pdf(pdf_bytes)

    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=selected_pdf,
        mime="application/pdf"
    )



