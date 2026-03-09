import time
import math
from html import escape
from pathlib import Path
from typing import List

import streamlit as st

from device_utils import get_device
from llm_utils import (
    analyze_candidate_profile,
    evaluate_answer,
    generate_custom_questions,
    generate_debrief_report,
    generate_followup_question,
    generate_interviewer_reply,
    generate_reference_answer,
    get_evaluator_backend_label,
    summarize_hiring_signal,
    warmup_evaluator_model,
)
from rag_engine import get_rag_engine, warmup_rag_system
from resume_parser import extract_text_from_upload


st.set_page_config(page_title="AI Interview Preparation Bot", page_icon=":dart:", layout="wide")

st.markdown(
    """
    <style>
        .hero { padding: 1rem 1.25rem; border-radius: 14px; background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; border: 1px solid #334155; margin-bottom: 1rem; }
        .question-card { padding: 1rem 1.1rem; border-radius: 12px; border: 1px solid #334155; background: #0b1220; margin: 0.5rem 0 1rem 0; }
        .question-source { font-size: 0.85rem; color: #93c5fd; margin-bottom: 0.4rem; }
        .summary-card { padding: 1rem 1.1rem; border-radius: 12px; border: 1px solid #1f2937; background: #0f172a; margin-top: 1rem; }
        .step-chip { display: inline-block; padding: 0.25rem 0.55rem; border-radius: 999px; border: 1px solid #334155; margin-right: 0.4rem; font-size: 0.8rem; color: #cbd5e1; background: #111827; }
        .loader-shell { margin-top: 2rem; padding: 1.2rem; border-radius: 16px; border: 1px solid #334155; background: linear-gradient(120deg, #020617 0%, #0f172a 100%); }
        .loader-bar { position: relative; height: 10px; border-radius: 999px; background: #1e293b; overflow: hidden; margin: 0.6rem 0 0.2rem 0; }
        .loader-bar::before { content: ""; position: absolute; inset: 0; width: 45%; background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8); animation: glide 1.2s ease-in-out infinite; }
        .runtime-chip { display:inline-block; padding: 0.25rem 0.6rem; border-radius: 999px; font-size: 0.78rem; border: 1px solid #334155; color:#cbd5e1; background:#111827; margin-left:0.5rem; }
        @keyframes glide { 0% { transform: translateX(-120%); } 50% { transform: translateX(70%); } 100% { transform: translateX(220%); } }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_FILE = Path(__file__).parent / "data" / "interview_questions.txt"
ROLES = ["Backend", "AI Engineer", "Data Scientist", "Fullstack"]
ROLE_MODES = ["Select Role", "Infer From Job Description"]
INTERVIEW_TONES = ["Easy", "Medium", "Strict"]
INTERVIEW_PERSONAS = ["Friendly Mentor", "FAANG Bar Raiser", "Startup CTO"]
TARGET_LEVELS = ["Intern", "Junior", "Mid", "Senior"]
COMPETENCY_DIMENSIONS = ["Technical Depth", "Communication", "System Thinking", "Practicality", "Trade-offs"]

LEVEL_BASE_EXPECTED = {"Intern": 4.8, "Junior": 5.8, "Mid": 6.8, "Senior": 7.8}
ROLE_EXPECTED_DELTA = {"Backend": 0.1, "AI Engineer": 0.2, "Data Scientist": 0.15, "Fullstack": 0.05}

ROLE_CATEGORY_HINTS = {
    "Backend": ["Backend Development", "System Design", "Software Engineering"],
    "AI Engineer": ["Machine Learning", "Data Science", "System Design"],
    "Data Scientist": ["Data Science", "Machine Learning"],
    "Fullstack": ["Software Engineering", "Backend Development", "System Design"],
}

ROLE_SKILL_HINTS = {
    "Backend": ["api", "backend", "database", "sql", "microservice", "docker", "kubernetes", "redis", "system design"],
    "AI Engineer": ["machine learning", "deep learning", "nlp", "pytorch", "tensorflow", "mlops", "llm", "vector"],
    "Data Scientist": ["data", "statistics", "python", "pandas", "numpy", "scikit", "experiment", "analytics"],
    "Fullstack": ["frontend", "backend", "react", "node", "javascript", "typescript", "api", "database", "system design"],
}

FULLSTACK_BLOCKLIST = {"machine learning", "deep learning", "nlp", "pytorch", "tensorflow", "mlops", "hadoop", "spark"}


for key, default in {
    "runtime_ready": False,
    "resume_text": "",
    "jd_text": "",
    "skills": [],
    "analysis_done": False,
    "analysis_role": "",
    "analysis_focus": "",
    "interview_role": "",
    "questions": [],
    "question_sources": [],
    "current_index": 0,
    "active_question_index": -1,
    "current_answer_input": "",
    "answers": [],
    "evaluations": [],
    "interview_started": False,
    "pending_retake": False,
    "pending_retake_role": "",
    "pending_retake_count": 6,
    "typed_done": {},
    "reference_cache": {},
    "interviewer_cache": {},
    "hiring_signal_cache": {},
    "debrief_cache": {},
    "followup_generated_for": [],
    "interview_tone": "Medium",
    "interviewer_persona": "Friendly Mentor",
    "target_level": "Mid",
    "role_mode": ROLE_MODES[0],
    "resume_file_sig": "",
    "jd_file_sig": "",
    "analysis_signature": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

evaluator_backend = get_evaluator_backend_label()


def _render_typing(
    container,
    text: str,
    anim_key: str,
    words_per_minute: int = 600,
    max_duration_sec: float = 12.0,
) -> None:
    text = (text or "").strip()
    if not text:
        return

    typed_done = st.session_state.get("typed_done", {})
    if typed_done.get(anim_key):
        container.write(text)
        return

    chars_per_second = max(1.0, (words_per_minute * 5) / 60.0)
    total_chars = len(text)
    chunk_size = 1
    estimated_duration = total_chars / chars_per_second
    if estimated_duration > max_duration_sec:
        chunk_size = max(1, int(total_chars / (chars_per_second * max_duration_sec)))
    delay = max(0.01, chunk_size / chars_per_second)

    for i in range(0, total_chars, chunk_size):
        container.write(text[: i + chunk_size])
        time.sleep(delay)

    typed_done[anim_key] = True
    st.session_state.typed_done = typed_done


def _uploaded_file_sig(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = getattr(uploaded_file, "name", "")
    size = getattr(uploaded_file, "size", 0)
    return f"{name}:{size}"


def _invalidate_profile_analysis() -> None:
    st.session_state.analysis_done = False
    st.session_state.analysis_role = ""
    st.session_state.analysis_focus = ""
    st.session_state.skills = []
    st.session_state.analysis_signature = ""


def _live_structure_hints(answer: str) -> List[str]:
    text = (answer or "").strip().lower()
    tokens = text.split()
    hints: List[str] = []
    if len(tokens) < 20:
        hints.append("Add more technical depth in 3-5 sentences.")
    if not any(k in text for k in ["example", "for instance", "e.g", "such as"]):
        hints.append("Include one concrete implementation example.")
    if not any(k in text for k in ["trade-off", "tradeoff", "pros", "cons", "however"]):
        hints.append("Mention one trade-off and why it matters.")
    if not any(ch.isdigit() for ch in text):
        hints.append("Add a measurable outcome or constraint (latency, scale, cost, etc.).")
    if not any(k in text for k in ["first", "then", "finally", "because"]):
        hints.append("Structure your answer clearly: approach -> reasoning -> result.")
    return hints[:3]


def _render_competency_radar(competencies: dict, chart_key: str, title: str) -> None:
    dims = COMPETENCY_DIMENSIONS
    values = [max(1, min(10, int(competencies.get(d, 1)))) for d in dims]
    n = len(dims)
    cx, cy, radius = 150, 150, 95

    def pt(angle: float, r: float) -> tuple[float, float]:
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    angles = [(-math.pi / 2) + (2 * math.pi * i / n) for i in range(n)]
    grid = []
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        pts = []
        for a in angles:
            x, y = pt(a, radius * frac)
            pts.append(f"{x:.1f},{y:.1f}")
        grid.append(f"<polygon points='{' '.join(pts)}' fill='none' stroke='#334155' stroke-width='1' />")

    axes = []
    labels = []
    for i, a in enumerate(angles):
        x, y = pt(a, radius)
        axes.append(f"<line x1='{cx}' y1='{cy}' x2='{x:.1f}' y2='{y:.1f}' stroke='#475569' stroke-width='1' />")
        lx, ly = pt(a, radius + 24)
        labels.append(
            f"<text x='{lx:.1f}' y='{ly:.1f}' fill='#cbd5e1' font-size='11' text-anchor='middle'>{escape(dims[i])}</text>"
        )

    data_pts = []
    for i, a in enumerate(angles):
        x, y = pt(a, radius * (values[i] / 10.0))
        data_pts.append(f"{x:.1f},{y:.1f}")

    svg = f"""
    <div style="border:1px solid #334155;border-radius:12px;padding:8px;background:#0b1220;">
      <div style="font-size:13px;color:#cbd5e1;margin:0 0 6px 4px;">{escape(title)}</div>
      <svg width="300" height="300" viewBox="0 0 300 300">
        {''.join(grid)}
        {''.join(axes)}
        <polygon points="{' '.join(data_pts)}" fill="rgba(56,189,248,0.28)" stroke="#38bdf8" stroke-width="2" />
        {''.join(labels)}
      </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)
    st.caption(" | ".join([f"{d}: {v}/10" for d, v in zip(dims, values)]))


def _percentile_benchmark(role: str, level: str, avg_score: float, competencies: dict) -> dict:
    base = LEVEL_BASE_EXPECTED.get(level, 6.8) + ROLE_EXPECTED_DELTA.get(role, 0.1)
    comp_avg = 0.0
    if competencies:
        comp_avg = sum(float(competencies.get(d, 0)) for d in COMPETENCY_DIMENSIONS) / max(1, len(COMPETENCY_DIMENSIONS))
    blended = (avg_score * 0.75) + (comp_avg * 0.25 if comp_avg else avg_score * 0.25)
    delta = blended - base
    percentile = int(max(5, min(99, round(50 + (delta * 18)))))
    if percentile >= 75:
        band = "Above Expected"
    elif percentile >= 45:
        band = "At Expected"
    else:
        band = "Below Expected"
    return {"percentile": percentile, "band": band, "expected_score": round(base, 2)}


def _filter_skills_for_role(skills: List[str], role: str) -> List[str]:
    role_hints = ROLE_SKILL_HINTS.get(role, [])
    filtered = []
    for skill in skills:
        s = skill.lower()
        if role == "Fullstack" and any(token in s for token in FULLSTACK_BLOCKLIST):
            continue
        if any(h in s for h in role_hints):
            filtered.append(skill)
    return filtered


def _start_interview(role: str, count: int, jd_context: str = "", persona: str = "Friendly Mentor") -> None:
    rag_engine = get_rag_engine(str(DATA_FILE))
    filtered_skills = _filter_skills_for_role(st.session_state.skills, role)
    skill_query = ", ".join(filtered_skills[:8])
    jd_query = (jd_context or "").strip()
    if len(jd_query) > 800:
        jd_query = jd_query[:800]

    query_parts = [f"Role: {role}"]
    if skill_query:
        query_parts.append(f"Skills: {skill_query}")
    if jd_query:
        query_parts.append(f"Job Description Context: {jd_query}")
    retrieval_query = ". ".join(query_parts)

    custom_count = max(1, count // 3)
    rag_count = max(1, count - custom_count)

    rag_questions = rag_engine.retrieve_questions(
        retrieval_query,
        top_k=rag_count,
        preferred_categories=ROLE_CATEGORY_HINTS.get(role, []),
    )
    custom_questions = generate_custom_questions(
        role=role,
        skills=filtered_skills,
        jd_context=jd_query,
        count=custom_count,
        persona=persona,
    )

    combined_questions: List[str] = []
    combined_sources: List[str] = []
    r, c = 0, 0
    while len(combined_questions) < count and (r < len(rag_questions) or c < len(custom_questions)):
        if r < len(rag_questions):
            q = rag_questions[r]
            r += 1
            if q not in combined_questions:
                combined_questions.append(q)
                combined_sources.append("RAG")
                if len(combined_questions) >= count:
                    break
        if c < len(custom_questions):
            q = custom_questions[c]
            c += 1
            if q not in combined_questions:
                combined_questions.append(q)
                combined_sources.append("CUSTOM")
                if len(combined_questions) >= count:
                    break

    if len(combined_questions) < count:
        fallback = rag_engine.retrieve_questions(
            retrieval_query,
            top_k=count,
            preferred_categories=ROLE_CATEGORY_HINTS.get(role, []),
        )
        for q in fallback:
            if q not in combined_questions:
                combined_questions.append(q)
                combined_sources.append("RAG")
            if len(combined_questions) >= count:
                break

    st.session_state.questions = combined_questions
    st.session_state.question_sources = combined_sources
    st.session_state.interview_role = role
    st.session_state.current_index = 0
    st.session_state.active_question_index = 0
    st.session_state.current_answer_input = ""
    st.session_state.answers = [""] * len(combined_questions)
    st.session_state.evaluations = [None] * len(combined_questions)
    st.session_state.interview_started = True
    st.session_state.typed_done = {}
    st.session_state.reference_cache = {}
    st.session_state.interviewer_cache = {}
    st.session_state.followup_generated_for = []


def _stop_interview() -> None:
    st.session_state.interview_started = False
    st.session_state.questions = []
    st.session_state.question_sources = []
    st.session_state.answers = []
    st.session_state.evaluations = []
    st.session_state.current_index = 0
    st.session_state.active_question_index = -1

    if "current_answer_input" in st.session_state:
        del st.session_state["current_answer_input"]

    st.session_state.interview_role = ""
    st.session_state.reference_cache = {}
    st.session_state.interviewer_cache = {}
    st.session_state.followup_generated_for = []


if not st.session_state.runtime_ready:
    st.markdown(
        """
        <div class="hero">
            <h2 style="margin:0;">Preparing AI Interview Bot</h2>
            <p style="margin:0.35rem 0 0 0;">Loading models and vector index. This happens once at startup.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="loader-shell">
            <div style="font-size:0.95rem;color:#cbd5e1;">Initializing runtime components...</div>
            <div class="loader-bar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress = st.progress(0)
    status = st.empty()
    steps = [
        ("Loading embedding model and building FAISS index...", lambda: warmup_rag_system(str(DATA_FILE))),
        (f"Loading evaluator backend: {evaluator_backend}...", warmup_evaluator_model),
    ]

    try:
        for i, (msg, fn) in enumerate(steps, start=1):
            status.info(msg)
            fn()
            progress.progress(int((i / len(steps)) * 100))
            time.sleep(0.15)
    except Exception as exc:
        st.error(f"Startup initialization failed: {exc}")
        st.stop()

    status.success("All models loaded. Launching interface...")
    time.sleep(0.2)
    st.session_state.runtime_ready = True
    st.rerun()

runtime_device = get_device().upper()
st.markdown(
    f"""
    <div class="hero">
        <h2 style="margin:0;">AI Interview Preparation Bot
            <span class="runtime-chip">Runtime: {runtime_device}</span>
            <span class="runtime-chip">Evaluator: {evaluator_backend}</span>
        </h2>
        <p style="margin:0.4rem 0 0 0;">Resume + JD aware role inference, tone-controlled interviewing, and AI-grounded evaluation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="step-chip">1. Resume + Optional JD</span>
    <span class="step-chip">2. AI Profile Analysis</span>
    <span class="step-chip">3. Retrieve Questions (RAG)</span>
    <span class="step-chip">4. Answer + Tone-based Evaluation</span>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.subheader("Workflow")
    role_mode = st.radio("Interview Setup", ROLE_MODES, key="role_mode", horizontal=True)
    manual_role = st.selectbox("Target Role", ROLES, key="manual_role", disabled=role_mode != ROLE_MODES[0])
    st.selectbox("Target Level", TARGET_LEVELS, key="target_level")
    question_count = st.slider("Number of Questions", min_value=3, max_value=12, value=6)
    uploaded_file = st.file_uploader("Step 1: Upload Resume (PDF/TXT)", type=["pdf", "txt"])
    jd_uploaded_file = None
    jd_pasted = ""
    if role_mode == ROLE_MODES[1]:
        jd_uploaded_file = st.file_uploader("Step 1b: Upload Job Description (PDF/TXT)", type=["pdf", "txt"])
        jd_pasted = st.text_area(
            "Or paste Job Description text",
            value=st.session_state.jd_text,
            height=120,
            placeholder="Paste JD text here if you don't upload a JD file...",
        )

    if uploaded_file is not None:
        resume_sig = _uploaded_file_sig(uploaded_file)
        if resume_sig != st.session_state.resume_file_sig:
            with st.spinner("Parsing resume..."):
                st.session_state.resume_text = extract_text_from_upload(uploaded_file)
            st.session_state.resume_file_sig = resume_sig
            _invalidate_profile_analysis()
    elif st.session_state.resume_file_sig:
        st.session_state.resume_text = ""
        st.session_state.resume_file_sig = ""
        _invalidate_profile_analysis()

    if role_mode == ROLE_MODES[1]:
        if jd_uploaded_file is not None:
            jd_sig = _uploaded_file_sig(jd_uploaded_file)
            if jd_sig != st.session_state.jd_file_sig:
                with st.spinner("Parsing job description..."):
                    st.session_state.jd_text = extract_text_from_upload(jd_uploaded_file)
                st.session_state.jd_file_sig = jd_sig
                _invalidate_profile_analysis()
        elif jd_pasted.strip() != st.session_state.jd_text.strip():
            st.session_state.jd_text = jd_pasted.strip()
            st.session_state.jd_file_sig = ""
            _invalidate_profile_analysis()
    else:
        # Hide JD input in manual-role mode and avoid using stale JD context.
        if st.session_state.jd_text:
            st.session_state.jd_text = ""
            st.session_state.jd_file_sig = ""
            _invalidate_profile_analysis()

    if st.button("Step 2: Analyze Role + Skills", use_container_width=True):
        if not st.session_state.resume_text:
            st.warning("Upload a resume first.")
        elif role_mode == ROLE_MODES[1] and not st.session_state.jd_text.strip():
            st.warning("Upload or paste a job description to infer role from JD.")
        else:
            selected_for_analysis = manual_role if role_mode == ROLE_MODES[0] else ""
            analysis_job_desc = st.session_state.jd_text if role_mode == ROLE_MODES[1] else ""
            analysis_sig = (
                f"{hash(st.session_state.resume_text)}::"
                f"{hash(analysis_job_desc)}::"
                f"{selected_for_analysis}::{role_mode}"
            )
            if st.session_state.analysis_done and st.session_state.analysis_signature == analysis_sig:
                st.info("Using existing analysis for current inputs.")
            else:
                with st.spinner("AI is analyzing resume and job context..."):
                    analysis = analyze_candidate_profile(
                        resume_text=st.session_state.resume_text,
                        job_description=analysis_job_desc,
                        selected_role=selected_for_analysis,
                    )
                st.session_state.skills = analysis.get("skills", [])
                st.session_state.analysis_role = analysis.get("role", manual_role)
                st.session_state.analysis_focus = analysis.get("jd_focus", "")
                st.session_state.analysis_done = True
                st.session_state.analysis_signature = analysis_sig

    if st.session_state.analysis_done:
        st.success("Profile analyzed")
        st.write(f"Role for interview: **{st.session_state.analysis_role or manual_role}**")
        if st.session_state.analysis_focus:
            _render_typing(
                st.empty(),
                f"JD focus: {st.session_state.analysis_focus}",
                f"focus_{hash(st.session_state.analysis_focus)}",
                words_per_minute=600,
            )

    if st.session_state.skills:
        skill_summary = ", ".join(st.session_state.skills)
        _render_typing(
            st.empty(),
            f"AI-detected skills: {skill_summary}",
            f"skills_{hash(skill_summary)}",
            words_per_minute=600,
        )

    if st.button("Step 3: Start Interview", use_container_width=True):
        if not st.session_state.analysis_done:
            st.warning("Run Step 2 first so AI can set interview role and skills.")
        else:
            role_for_interview = st.session_state.analysis_role or manual_role
            jd_context = st.session_state.jd_text if role_mode == ROLE_MODES[1] else ""
            _start_interview(
                role_for_interview,
                question_count,
                jd_context=jd_context,
                persona=st.session_state.interviewer_persona,
            )
            st.rerun()

selected_role = st.session_state.analysis_role or st.session_state.get("manual_role", ROLES[0])

if st.session_state.pending_retake:
    role = st.session_state.pending_retake_role or selected_role
    count = int(st.session_state.pending_retake_count or question_count)
    st.session_state.pending_retake = False
    jd_context = st.session_state.jd_text if st.session_state.role_mode == ROLE_MODES[1] else ""
    _start_interview(
        role,
        count,
        jd_context=jd_context,
        persona=st.session_state.interviewer_persona,
    )
    st.rerun()

with right_col:
    st.subheader("Interview Panel")
    st.select_slider("Interviewer Tone", options=INTERVIEW_TONES, key="interview_tone")
    st.selectbox("Interviewer Persona", INTERVIEW_PERSONAS, key="interviewer_persona")
    tone_value = (st.session_state.interview_tone or "Medium").strip().lower()
    persona_value = st.session_state.interviewer_persona
    interview_role = st.session_state.interview_role or selected_role

    if not st.session_state.interview_started:
        st.info("Complete steps on the left to begin interview mode.")
    else:
        questions = st.session_state.questions
        sources = st.session_state.question_sources
        idx = st.session_state.current_index

        if st.session_state.active_question_index != idx:
            st.session_state.current_answer_input = st.session_state.answers[idx]
            st.session_state.active_question_index = idx

        st.write(f"Question {idx + 1} of {len(questions)}")
        source_label = sources[idx] if idx < len(sources) else "RAG"

        st.markdown(
            f"""
            <div class="question-card">
                <div class="question-source">Source: [{source_label}]</div>
                <div class="question-source">Interviewer tone: [{st.session_state.interview_tone}]</div>
                <div class="question-source">Interviewer persona: [{persona_value}]</div>
                <div><strong>{questions[idx]}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_area(
            "Step 4: Your Answer",
            key="current_answer_input",
            height=160,
            placeholder="Explain your approach clearly with technical depth and an example...",
        )
        hints = _live_structure_hints(st.session_state.current_answer_input)
        if hints:
            st.caption("Live Coach")
            for h in hints:
                st.caption(f"- {h}")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Submit Answer", use_container_width=True):
                st.session_state.answers[idx] = st.session_state.current_answer_input
                question_key = f"{interview_role}::{tone_value}::{persona_value}::{questions[idx]}"
                with st.spinner(
                    f"AI interviewer reviewing your answer and preparing reference via {evaluator_backend}..."
                ):
                    evaluation = evaluate_answer(
                        questions[idx],
                        st.session_state.current_answer_input,
                        interview_role,
                        tone=tone_value,
                        persona=persona_value,
                    )
                    st.session_state.interviewer_cache[question_key] = generate_interviewer_reply(
                        question=questions[idx],
                        answer=st.session_state.current_answer_input,
                        role=interview_role,
                        tone=tone_value,
                        persona=persona_value,
                        score=evaluation["score"],
                        strengths=evaluation["strengths"],
                        improvements=evaluation["improvements"],
                    )
                    st.session_state.reference_cache[question_key] = generate_reference_answer(
                        questions[idx],
                        interview_role,
                        tone=tone_value,
                        persona=persona_value,
                    )
                st.session_state.evaluations[idx] = evaluation

                # Adaptive follow-up generation: deepen weak answers, stretch strong answers.
                followup_key = f"{idx}:{questions[idx]}"
                seen_followups = set(st.session_state.followup_generated_for or [])
                source_now = sources[idx] if idx < len(sources) else "RAG"
                if "FOLLOW-UP" not in source_now and followup_key not in seen_followups:
                    followup = generate_followup_question(
                        question=questions[idx],
                        answer=st.session_state.current_answer_input,
                        role=interview_role,
                        score=int(evaluation.get("score", 1)),
                        tone=tone_value,
                        persona=persona_value,
                    )
                    if followup and followup.get("question"):
                        fq = followup["question"].strip()
                        if fq and fq not in st.session_state.questions:
                            insert_at = idx + 1
                            st.session_state.questions.insert(insert_at, fq)
                            st.session_state.question_sources.insert(insert_at, followup.get("source", "FOLLOW-UP"))
                            st.session_state.answers.insert(insert_at, "")
                            st.session_state.evaluations.insert(insert_at, None)
                    seen_followups.add(followup_key)
                    st.session_state.followup_generated_for = list(seen_followups)
                st.rerun()

        with c2:
            if st.button("Next Question", use_container_width=True):
                st.session_state.answers[idx] = st.session_state.current_answer_input
                if idx < len(questions) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
                else:
                    st.success("Interview completed. Scroll down for summary.")

        with c3:
            if st.button("Stop Interview", use_container_width=True):

                completed = [x for x in st.session_state.evaluations if x is not None]

                if completed:
                    signal = summarize_hiring_signal(
                        role=interview_role,
                        level=st.session_state.target_level,
                        tone=tone_value,
                        persona=persona_value,
                        questions=st.session_state.questions,
                        answers=st.session_state.answers,
                        evaluations=st.session_state.evaluations,
                    )

                    report = generate_debrief_report(
                        role=interview_role,
                        level=st.session_state.target_level,
                        tone=tone_value,
                        persona=persona_value,
                        questions=st.session_state.questions,
                        answers=st.session_state.answers,
                        evaluations=st.session_state.evaluations,
                    )

                    st.download_button(
                        "Download Interview Report (.md)",
                        data=report,
                        file_name=f"interview_debrief_{interview_role.lower().replace(' ', '_')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

                _stop_interview()
                st.rerun()

        if st.session_state.evaluations[idx] is not None:
            ev = st.session_state.evaluations[idx]
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Score", f"{ev['score']}/10")
            with m2:
                st.caption("Strengths")
                _render_typing(st.empty(), ev["strengths"], f"q{idx}_strengths_{hash(ev['strengths'])}", words_per_minute=600)
            with m3:
                st.caption("Weaknesses / Missing Points")
                _render_typing(
                    st.empty(),
                    ev["improvements"],
                    f"q{idx}_improvements_{hash(ev['improvements'])}",
                    words_per_minute=600,
                )

            competencies = ev.get("competencies", {})
            if competencies:
                _render_competency_radar(
                    competencies=competencies,
                    chart_key=f"q{idx}_radar",
                    title="Competency Radar (This Answer)",
                )

            question_key = f"{interview_role}::{tone_value}::{persona_value}::{questions[idx]}"
            interviewer_reply = st.session_state.interviewer_cache.get(question_key, "")
            if interviewer_reply:
                st.caption("AI Interviewer")
                _render_typing(
                    st.empty(),
                    interviewer_reply,
                    f"q{idx}_interviewer_{hash(interviewer_reply)}",
                    words_per_minute=600,
                )

            with st.expander("AI Reference Answer (Learn from this)", expanded=False):
                ref = st.session_state.reference_cache.get(question_key, "")
                if ref:
                    _render_typing(st.empty(), ref, f"q{idx}_model_{hash(ref)}", words_per_minute=600)
                else:
                    st.caption("Reference answer will appear after you submit this question.")

        completed = [x for x in st.session_state.evaluations if x is not None]
        if completed:
            avg = sum(x["score"] for x in completed) / len(completed)
            avg_comp = {}
            for dim in COMPETENCY_DIMENSIONS:
                vals = [float(ev.get("competencies", {}).get(dim, 0)) for ev in completed if ev.get("competencies")]
                if vals:
                    avg_comp[dim] = round(sum(vals) / len(vals), 2)
            benchmark = _percentile_benchmark(interview_role, st.session_state.target_level, avg, avg_comp)

            st.markdown(
                f"""
                <div class="summary-card">
                    <h4 style="margin:0 0 0.5rem 0;">Final Interview Summary</h4>
                    <p style="margin:0.2rem 0;">Role: <strong>{interview_role}</strong></p>
                    <p style="margin:0.2rem 0;">Level: <strong>{st.session_state.target_level}</strong></p>
                    <p style="margin:0.2rem 0;">Interviewer: <strong>{persona_value}</strong> ({st.session_state.interview_tone})</p>
                    <p style="margin:0.2rem 0;">Questions Answered: <strong>{len(completed)}/{len(questions)}</strong></p>
                    <p style="margin:0.2rem 0;">Current Average Score: <strong>{avg:.2f}/10</strong></p>
                    <p style="margin:0.2rem 0;">Benchmark: <strong>{benchmark['percentile']}th percentile</strong> ({benchmark['band']})</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if avg_comp:
                _render_competency_radar(
                    competencies=avg_comp,
                    chart_key="final_competency_radar",
                    title="Competency Radar (Overall)",
                )

            # Real hiring signal and one-click debrief when the interview round is complete.
            if len(completed) == len(questions):
                signal_key = (
                    f"{interview_role}|{st.session_state.target_level}|{tone_value}|{persona_value}|"
                    f"{hash(tuple(st.session_state.questions))}|{hash(tuple(st.session_state.answers))}|"
                    f"{hash(tuple(int(e.get('score', 1)) for e in completed))}"
                )
                if signal_key not in st.session_state.hiring_signal_cache:
                    with st.spinner("Generating hiring signal and debrief report..."):
                        signal = summarize_hiring_signal(
                            role=interview_role,
                            level=st.session_state.target_level,
                            tone=tone_value,
                            persona=persona_value,
                            questions=st.session_state.questions,
                            answers=st.session_state.answers,
                            evaluations=st.session_state.evaluations,
                        )
                        report = generate_debrief_report(
                            role=interview_role,
                            level=st.session_state.target_level,
                            tone=tone_value,
                            persona=persona_value,
                            questions=st.session_state.questions,
                            answers=st.session_state.answers,
                            evaluations=st.session_state.evaluations,
                        )
                    st.session_state.hiring_signal_cache[signal_key] = signal
                    st.session_state.debrief_cache[signal_key] = report

                hiring = st.session_state.hiring_signal_cache.get(signal_key, {})
                if hiring:
                    st.markdown(
                        f"""
                        <div class="summary-card">
                            <h4 style="margin:0 0 0.5rem 0;">Hiring Signal</h4>
                            <p style="margin:0.2rem 0;">Decision: <strong>{hiring.get('decision', 'NO_HIRE')}</strong></p>
                            <p style="margin:0.2rem 0;">{hiring.get('rationale', '')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    evidence = hiring.get("evidence", [])
                    if isinstance(evidence, list):
                        for line in evidence[:4]:
                            st.caption(f"- {line}")

                report_md = st.session_state.debrief_cache.get(signal_key, "")
                if report_md:
                    st.download_button(
                        "Download Debrief Report (.md)",
                        data=report_md,
                        file_name=f"interview_debrief_{interview_role.lower().replace(' ', '_')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

            if st.button("Retake Interview", use_container_width=True):
                st.session_state.pending_retake = True
                st.session_state.pending_retake_role = interview_role
                st.session_state.pending_retake_count = question_count
                st.rerun()
