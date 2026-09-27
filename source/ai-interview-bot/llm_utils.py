import json
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import pipeline

from device_utils import get_device, get_pipeline_device


DEFAULT_EVALUATOR_PROVIDER = os.getenv("EVALUATOR_PROVIDER", "local").strip().lower()
DEFAULT_EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_FALLBACKS = ["Qwen/Qwen2.5-1.5B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]
_CONFIG_PATH = Path(__file__).parent / "config.json"
NIM_API_KEY = ""
if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, "r") as f:
        NIM_API_KEY = json.load(f).get("NIM_API_KEY", "")
else:
    NIM_API_KEY = os.getenv("NIM_API_KEY", "")

DEFAULT_NIM_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.2-11b-vision-instruct")
KNOWN_ROLES = ["Backend", "AI Engineer", "Data Scientist", "Fullstack"]
TONE_LABELS = {"easy": "Easy", "medium": "Medium", "strict": "Strict"}
TONE_SCORE_BIAS = {"easy": 1, "medium": 0, "strict": -1}
PERSONA_INSTRUCTIONS = {
    "Friendly Mentor": "Warm, encouraging, and coaching-oriented while still technically accurate.",
    "FAANG Bar Raiser": "Highly rigorous, depth-focused, and strict about trade-offs and scale.",
    "Startup CTO": "Pragmatic, product-minded, and focused on speed, ownership, and impact.",
}
COMPETENCY_DIMENSIONS = [
    "Technical Depth",
    "Communication",
    "System Thinking",
    "Practicality",
    "Trade-offs",
]


def get_evaluator_provider() -> str:
    provider = DEFAULT_EVALUATOR_PROVIDER
    if provider in {"puter", "puter-gemini", "gemini", "nim"}:
        return "nim"
    return "local"


def get_evaluator_backend_label() -> str:
    if get_evaluator_provider() == "nim":
        return f"Nvidia NIM ({DEFAULT_NIM_MODEL})"
    return f"Local HF ({DEFAULT_EVALUATOR_MODEL})"


def _candidate_models() -> List[str]:
    candidates = [DEFAULT_EVALUATOR_MODEL]
    for name in MODEL_FALLBACKS:
        if name not in candidates:
            candidates.append(name)
    return candidates


def _build_evaluator(device_arg, model_name: str):
    return pipeline("text-generation", model=model_name, device=device_arg)


@lru_cache(maxsize=1)
def _get_local_evaluator_with_name() -> Tuple[Any, str]:
    models = _candidate_models()
    if get_device() == "mps":
        for model_name in models:
            try:
                return _build_evaluator(get_pipeline_device(), model_name), model_name
            except Exception:
                continue
    for model_name in models:
        try:
            return _build_evaluator(-1, model_name), model_name
        except Exception:
            continue
    raise RuntimeError("Failed to load local evaluator model.")


def _get_local_evaluator():
    evaluator, _ = _get_local_evaluator_with_name()
    return evaluator


def _extract_generated_text(output_item: Dict[str, Any], prompt: str) -> str:
    generated = output_item.get("generated_text")
    if isinstance(generated, list):
        if generated and isinstance(generated[-1], dict):
            return str(generated[-1].get("content") or generated[-1].get("text") or generated[-1]).strip()
        return " ".join(str(item) for item in generated).strip()
    if isinstance(generated, str):
        text = generated.strip()
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text
    return str(output_item).strip()


def _run_local_chat(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output_item = _get_local_evaluator()(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.08,
        no_repeat_ngram_size=3,
        return_full_text=False,
    )[0]
    return _extract_generated_text(output_item, user_prompt)


import urllib.request
import urllib.error

def _run_nim_chat(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEFAULT_NIM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.5
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        raise RuntimeError(f"NIM chat failed: HTTP {e.code} - {error_msg}")
    except Exception as e:
        raise RuntimeError(f"NIM chat failed: {e}")


def _run_chat(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    if get_evaluator_provider() == "nim":
        return _run_nim_chat(system_prompt, user_prompt, max_new_tokens)
    return _run_local_chat(system_prompt, user_prompt, max_new_tokens)


def warmup_evaluator_model() -> None:
    _run_chat(
        system_prompt="You are a helpful assistant.",
        user_prompt="Reply with one word: warm.",
        max_new_tokens=8,
    )


def _normalize_line(text: Any) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(r"^(strengths?|weaknesses?|improvements?|score)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _coerce_score(value: Any) -> int:
    if isinstance(value, int):
        return max(1, min(10, value))
    match = re.search(r"\b(10|[1-9])\b", str(value or ""))
    if not match:
        return 1
    return int(match.group(1))


def _normalize_tone(tone: str) -> str:
    t = (tone or "").strip().lower()
    if t in TONE_LABELS:
        return t
    return "medium"


def _normalize_persona(persona: str) -> str:
    p = (persona or "").strip()
    if p in PERSONA_INSTRUCTIONS:
        return p
    return "Friendly Mentor"


def _normalize_role_name(role_value: Any, fallback: str = "Backend") -> str:
    raw = str(role_value or "").strip().lower()
    if not raw:
        return fallback
    alias_map = {
        "backend": "Backend",
        "backend engineer": "Backend",
        "software engineer": "Backend",
        "ai engineer": "AI Engineer",
        "ml engineer": "AI Engineer",
        "machine learning engineer": "AI Engineer",
        "data scientist": "Data Scientist",
        "data science": "Data Scientist",
        "fullstack": "Fullstack",
        "full stack": "Fullstack",
        "full-stack": "Fullstack",
    }
    if raw in alias_map:
        return alias_map[raw]
    for known in KNOWN_ROLES:
        if known.lower() in raw:
            return known
    return fallback


def _coerce_skill_list(value: Any, top_k: int = 12) -> List[str]:
    items: List[str] = []
    if isinstance(value, list):
        items = [str(x).strip() for x in value]
    elif isinstance(value, str):
        items = [x.strip() for x in re.split(r"[,;\n]+", value)]

    cleaned: List[str] = []
    seen = set()
    for item in items:
        skill = re.sub(r"\s+", " ", item).strip().lower()
        if not skill or len(skill) < 2:
            continue
        if skill in seen:
            continue
        seen.add(skill)
        cleaned.append(skill)
        if len(cleaned) >= top_k:
            break
    return cleaned


def _heuristic_role_from_text(job_desc: str, resume_text: str, fallback: str = "Backend") -> str:
    text = f"{job_desc}\n{resume_text}".lower()
    if any(k in text for k in ["pytorch", "tensorflow", "llm", "nlp", "mlops"]):
        return "AI Engineer"
    if any(k in text for k in ["hypothesis", "statistics", "ab test", "eda", "regression"]):
        return "Data Scientist"
    if any(k in text for k in ["react", "frontend", "full stack", "typescript", "next.js"]):
        return "Fullstack"
    return fallback


def _heuristic_skills(text: str, top_k: int = 12) -> List[str]:
    candidates = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+.#/\-]{1,24}\b", text or "")
    preferred = {
        "python", "java", "go", "rust", "c++", "pytorch", "tensorflow", "sql", "postgres",
        "mongodb", "redis", "docker", "kubernetes", "aws", "gcp", "azure", "spark", "hadoop",
        "react", "node", "node.js", "typescript", "javascript", "fastapi", "django", "flask",
        "llm", "mlops", "airflow", "kafka", "faiss", "numpy", "pandas", "scikit", "xgboost",
    }
    out: List[str] = []
    seen = set()
    for token in candidates:
        t = token.lower()
        if t not in preferred or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= top_k:
            break
    return out


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]*", (text or "").lower())


def _is_low_signal_answer(answer: str) -> bool:
    tokens = _tokenize(answer)
    low_signal = {"idk", "bro", "whatever", "nothing", "no", "don't", "dont", "know", "god"}
    return len(tokens) <= 5 and any(tok in low_signal for tok in tokens)


def _is_profane(answer: str) -> bool:
    tokens = set(_tokenize(answer))
    banned = {
        "fuck", "fucking", "fucked", "shit", "bitch", "asshole", "dick", "cock", "pussy",
        "motherfucker", "bastard", "slut", "whore",
    }
    return bool(tokens.intersection(banned))


def _is_relevant_to_question(question: str, answer: str) -> bool:
    q_tokens = set(_tokenize(question))
    a_tokens = set(_tokenize(answer))
    if not a_tokens:
        return False
    overlap = len(q_tokens.intersection(a_tokens))
    return overlap >= 1


def _technical_richness(answer: str) -> int:
    tokens = set(_tokenize(answer))
    markers = {
        "missing", "outliers", "correlation", "correlations", "distribution", "distributions",
        "leakage", "regularization", "feature", "features", "scalable", "queue", "scheduler",
        "latency", "throughput", "monitoring", "logging", "retry", "retries", "idempotent",
        "docker", "kubernetes", "pytorch", "tensorflow", "sql", "redis", "cache",
        "histogram", "histograms", "boxplot", "boxplots", "describe", "dataset", "eda",
    }
    return len(tokens.intersection(markers))


def _heuristic_score(question: str, answer: str) -> int:
    q_tokens = set(_tokenize(question))
    a_tokens = _tokenize(answer)
    if not a_tokens:
        return 1

    overlap = len(q_tokens.intersection(set(a_tokens)))
    length = len(a_tokens)
    score = 1
    if length >= 4:
        score = 2
    if length >= 10:
        score = 3
    if overlap >= 1:
        score = max(score, 4)
    if overlap >= 3:
        score = max(score, 6)
    if overlap >= 6:
        score = max(score, 7)
    if length >= 25 and overlap >= 3:
        score += 1
    if length >= 45 and overlap >= 5:
        score += 1

    richness = _technical_richness(answer)
    if richness >= 3:
        score = max(score, 6)
    if richness >= 5 and length >= 20:
        score = max(score, 7)
    if richness >= 6 and overlap >= 1 and length >= 28:
        score = max(score, 8)

    if _is_low_signal_answer(answer):
        return 1
    if _is_profane(answer):
        return 1
    if not _is_relevant_to_question(question, answer):
        return min(score, 2)

    return max(1, min(10, score))


def _top_grounded_phrases(question: str, answer: str, limit: int = 2) -> List[str]:
    q_tokens = set(_tokenize(question))
    phrases = [p.strip() for p in re.split(r"[\n,;]+", answer or "") if p.strip()]
    ranked: List[Tuple[int, str]] = []
    for phrase in phrases:
        tokens = set(_tokenize(phrase))
        overlap = len(tokens.intersection(q_tokens))
        if overlap > 0:
            ranked.append((overlap, phrase))
    ranked.sort(key=lambda x: (-x[0], -len(x[1])))
    return [p for _, p in ranked[:limit]]


def _derive_strength(question: str, answer: str) -> str:
    if _is_profane(answer) or _is_low_signal_answer(answer) or not _is_relevant_to_question(question, answer):
        return "No clear technical strength shown."

    grounded = _top_grounded_phrases(question, answer, limit=2)
    if not grounded:
        return "No clear technical strength shown."
    if len(grounded) == 1:
        return f'You correctly mentioned "{grounded[0]}", which is relevant to the question.'
    return f'You covered relevant points such as "{grounded[0]}" and "{grounded[1]}".'


def _derive_weakness(question: str, answer: str) -> str:
    a_tokens = _tokenize(answer)
    q_tokens = set(_tokenize(question))
    overlap = len(q_tokens.intersection(set(a_tokens)))
    answer_l = (answer or "").lower()
    richness = _technical_richness(answer)

    missing: List[str] = []
    if len(a_tokens) < 12:
        missing.append("add more technical depth")
    if overlap < 2 and richness < 3:
        missing.append("align the response more directly with the asked question")
    if not any(k in answer_l for k in ["because", "therefore", "so that", "trade-off", "tradeoff", "pros", "cons"]):
        missing.append("explain why your approach works and include a trade-off")
    if not any(k in answer_l for k in ["example", "for instance", "e.g", "such as"]):
        missing.append("include one concrete example")

    if _is_profane(answer):
        return "The answer is unprofessional and not technically relevant; provide a respectful, technical response."
    if not missing:
        return "Add implementation details on scale, reliability, and monitoring to make the answer stronger."
    return "You should " + ", and ".join(missing[:2]) + "."


def _clamp_score(x: int) -> int:
    return max(1, min(10, int(x)))


def _heuristic_competencies(question: str, answer: str, overall_score: int) -> Dict[str, int]:
    tokens = _tokenize(answer)
    length = len(tokens)
    richness = _technical_richness(answer)
    answer_l = (answer or "").lower()
    relevance = 1 if _is_relevant_to_question(question, answer) else 0

    technical = _clamp_score(max(overall_score - 1, 2) + (1 if richness >= 4 else 0))
    communication = _clamp_score(
        3
        + (1 if length >= 12 else 0)
        + (1 if length >= 25 else 0)
        + (1 if any(w in answer_l for w in ["first", "then", "finally", "because"]) else 0)
    )
    system_thinking = _clamp_score(
        2
        + (2 if any(w in answer_l for w in ["scale", "latency", "throughput", "reliability", "monitoring"]) else 0)
        + (1 if "distributed" in answer_l or "queue" in answer_l else 0)
        + (1 if relevance else 0)
    )
    practicality = _clamp_score(
        2
        + (2 if any(w in answer_l for w in ["for example", "example", "e.g", "such as"]) else 0)
        + (1 if any(w in answer_l for w in ["implemented", "used", "built", "deployed"]) else 0)
        + (1 if any(ch.isdigit() for ch in answer_l) else 0)
    )
    tradeoffs = _clamp_score(
        1
        + (3 if any(w in answer_l for w in ["trade-off", "tradeoff", "pros", "cons", "however"]) else 0)
        + (1 if "because" in answer_l else 0)
    )

    return {
        "Technical Depth": technical,
        "Communication": communication,
        "System Thinking": system_thinking,
        "Practicality": practicality,
        "Trade-offs": tradeoffs,
    }


def _sanitize_competencies(raw: Any, fallback: Dict[str, int]) -> Dict[str, int]:
    out = {}
    if isinstance(raw, dict):
        for dim in COMPETENCY_DIMENSIONS:
            value = raw.get(dim)
            if value is None:
                # Support lowercase keys from model output.
                value = raw.get(dim.lower())
            if value is None:
                value = fallback[dim]
            out[dim] = _coerce_score(value)
    else:
        out = fallback.copy()
    for dim in COMPETENCY_DIMENSIONS:
        out.setdefault(dim, fallback[dim])
        out[dim] = _clamp_score(out[dim])
    return out


def _score_competencies_with_model(
    question: str,
    answer: str,
    role: str,
    tone: str,
    persona: str,
    overall_score: int,
) -> Optional[Dict[str, int]]:
    tone_value = _normalize_tone(tone)
    persona_value = _normalize_persona(persona)
    prompt = (
        "Score the candidate answer on five competencies.\n"
        "Return strict JSON only with integer values from 1 to 10.\n"
        "Keys must be exactly: Technical Depth, Communication, System Thinking, Practicality, Trade-offs.\n\n"
        f"Role: {role}\n"
        f"Interviewer tone: {TONE_LABELS[tone_value]}\n"
        f"Interviewer persona: {persona_value} ({PERSONA_INSTRUCTIONS[persona_value]})\n"
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n"
        f"Overall score context: {overall_score}/10\n"
    )
    raw = _run_chat(
        system_prompt="You are a precise interview rubric grader.",
        user_prompt=prompt,
        max_new_tokens=180,
    )
    parsed = _extract_json_block(raw)
    if not parsed:
        return None
    base = _heuristic_competencies(question, answer, overall_score)
    scored = _sanitize_competencies(parsed, base)
    # Keep competency values consistent with overall score envelope.
    for dim in COMPETENCY_DIMENSIONS:
        scored[dim] = _clamp_score(max(overall_score - 3, min(overall_score + 3, scored[dim])))
    return scored


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.IGNORECASE | re.MULTILINE).strip()
    candidates = [raw]
    for match in re.finditer(r"\{[\s\S]*?\}", raw):
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def analyze_candidate_profile(resume_text: str, job_description: str, selected_role: str = "") -> Dict[str, Any]:
    safe_resume = (resume_text or "").strip()
    safe_jd = (job_description or "").strip()
    manual_role = _normalize_role_name(selected_role, fallback="Backend") if selected_role else ""
    fallback_role = manual_role or _heuristic_role_from_text(safe_jd, safe_resume, fallback="Backend")

    prompt = (
        "Analyze the candidate profile for technical interview preparation.\n"
        "Use resume and job description context to infer the best interview role and strongest technical skills.\n"
        "Return strict JSON only with keys: role, skills, jd_focus.\n"
        "- role must be one of: Backend, AI Engineer, Data Scientist, Fullstack\n"
        "- skills must be a list of 6 to 12 lowercase technical skills relevant to the role\n"
        "- jd_focus must be one short sentence summarizing what the job description emphasizes\n\n"
        f"Manual role override (empty means infer from JD): {selected_role}\n"
        f"Job Description:\n{safe_jd or '[none provided]'}\n\n"
        f"Resume:\n{safe_resume[:6000]}"
    )

    parsed: Optional[Dict[str, Any]] = None
    try:
        raw = _run_chat(
            system_prompt="You are an expert technical recruiter and hiring manager.",
            user_prompt=prompt,
            max_new_tokens=300,
        )
        parsed = _extract_json_block(raw)
    except Exception:
        parsed = None

    if parsed:
        role = _normalize_role_name(parsed.get("role"), fallback=fallback_role)
        if selected_role:
            role = manual_role
        skills = _coerce_skill_list(parsed.get("skills"), top_k=12)
        if not skills:
            skills = _heuristic_skills(f"{safe_resume}\n{safe_jd}", top_k=12)
        jd_focus = _normalize_line(parsed.get("jd_focus") or "Interview focus generated from resume and role context.")
        return {"role": role, "skills": skills, "jd_focus": jd_focus}

    return {
        "role": fallback_role,
        "skills": _heuristic_skills(f"{safe_resume}\n{safe_jd}", top_k=12),
        "jd_focus": "Interview focus inferred from available profile context.",
    }


def _looks_grounded_feedback(feedback: str, question: str, answer: str) -> bool:
    text = (feedback or "").strip().lower()
    if not text:
        return False
    if "no clear technical strength" in text:
        return True
    f_tokens = set(_tokenize(feedback))
    a_tokens = set(_tokenize(answer))
    q_tokens = set(_tokenize(question))
    return len(f_tokens.intersection(a_tokens.union(q_tokens))) >= 2


def _evaluate_with_model(
    question: str,
    answer: str,
    role: str,
    tone: str,
    persona: str,
) -> Optional[Dict[str, str]]:
    tone_value = _normalize_tone(tone)
    persona_value = _normalize_persona(persona)
    tone_instruction = {
        "easy": "Be supportive and coaching-oriented while staying truthful to the answer.",
        "medium": "Be balanced and interview-realistic.",
        "strict": "Be demanding and rigorous; expect depth, trade-offs, and concrete technical detail.",
    }[tone_value]

    prompt = (
        "Evaluate the candidate answer for this interview question.\n"
        "Use only evidence present in the candidate answer. Do not invent claims.\n"
        "Return JSON only with keys: score, strengths, improvements.\n"
        "Rules:\n"
        "- score: integer 1 to 10\n"
        "- strengths: one concise sentence about what the candidate actually did well\n"
        "- improvements: one concise sentence on what is missing or can be improved\n"
        "- If answer is abusive/off-topic/very low-quality, set score <= 2 and reflect that clearly.\n\n"
        f"Interviewer tone: {TONE_LABELS[tone_value]} ({tone_instruction})\n"
        f"Interviewer persona: {persona_value} ({PERSONA_INSTRUCTIONS[persona_value]})\n"
        f"Role: {role}\n"
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n"
    )
    raw = _run_chat(
        system_prompt="You are a strict technical interviewer and accurate grader.",
        user_prompt=prompt,
        max_new_tokens=220,
    )
    parsed = _extract_json_block(raw)
    if not parsed:
        return None
    return {
        "score": _coerce_score(parsed.get("score")),
        "strengths": _normalize_line(parsed.get("strengths")),
        "improvements": _normalize_line(parsed.get("improvements")),
    }


def _final_score(
    question: str,
    answer: str,
    strength: str,
    tone: str,
    model_score: Optional[int] = None,
) -> int:
    tone_value = _normalize_tone(tone)
    heuristic = _heuristic_score(question, answer)
    score = heuristic if model_score is None else model_score
    if model_score is not None:
        score = max(heuristic - 2, min(heuristic + 2, model_score))
    score += TONE_SCORE_BIAS[tone_value]
    if strength.lower().startswith("no clear technical strength"):
        score = min(score, 4)
    if _is_profane(answer):
        score = 1
    if _is_low_signal_answer(answer):
        score = min(score, 2)
    return max(1, min(10, score))


def evaluate_answer(
    question: str,
    answer: str,
    role: str,
    tone: str = "medium",
    persona: str = "Friendly Mentor",
) -> Dict[str, Any]:
    safe_answer = (answer or "").strip() or "No candidate answer provided."
    tone_value = _normalize_tone(tone)
    persona_value = _normalize_persona(persona)

    heuristic_strength = _normalize_line(_derive_strength(question, safe_answer))
    heuristic_weakness = _normalize_line(_derive_weakness(question, safe_answer))

    model_eval: Optional[Dict[str, str]] = None
    try:
        model_eval = _evaluate_with_model(
            question,
            safe_answer,
            role,
            tone=tone_value,
            persona=persona_value,
        )
    except Exception:
        model_eval = None

    strengths = heuristic_strength
    improvements = heuristic_weakness
    model_score: Optional[int] = None

    if model_eval:
        maybe_strength = _normalize_line(model_eval.get("strengths", ""))
        maybe_improvements = _normalize_line(model_eval.get("improvements", ""))

        if maybe_strength and _looks_grounded_feedback(maybe_strength, question, safe_answer):
            strengths = maybe_strength
        if maybe_improvements and _looks_grounded_feedback(maybe_improvements, question, safe_answer):
            improvements = maybe_improvements

        model_score = _coerce_score(model_eval.get("score"))

    final_score = _final_score(question, safe_answer, strengths, tone=tone_value, model_score=model_score)
    competencies = _heuristic_competencies(question, safe_answer, final_score)
    try:
        modeled_comp = _score_competencies_with_model(
            question=question,
            answer=safe_answer,
            role=role,
            tone=tone_value,
            persona=persona_value,
            overall_score=final_score,
        )
        if modeled_comp:
            competencies = modeled_comp
    except Exception:
        pass

    evidence = _top_grounded_phrases(question, safe_answer, limit=3)
    return {
        "score": final_score,
        "strengths": strengths,
        "improvements": improvements,
        "competencies": competencies,
        "evidence": evidence,
    }


@lru_cache(maxsize=512)
def generate_reference_answer(
    question: str,
    role: str,
    tone: str = "medium",
    persona: str = "Friendly Mentor",
) -> str:
    tone_value = _normalize_tone(tone)
    persona_value = _normalize_persona(persona)
    tone_instruction = {
        "easy": "Keep wording approachable but still technical.",
        "medium": "Keep wording balanced and interview-ready.",
        "strict": "Make it highly rigorous with implementation specifics and trade-offs.",
    }[tone_value]

    answer = _run_chat(
        system_prompt="You are a candidate in a technical interview. Answer directly and clearly.",
        user_prompt=(
            f"Role: {role}\n"
            f"Interviewer tone: {TONE_LABELS[tone_value]} ({tone_instruction})\n"
            f"Interviewer persona: {persona_value} ({PERSONA_INSTRUCTIONS[persona_value]})\n"
            f"Question: {question}\n\n"
            "Provide the best candidate answer in 5 to 7 complete sentences.\n"
            "Write in first-person interview style, technically detailed, coherent, and concise.\n"
            "Do not provide coaching tips, headings, or bullet points."
        ),
        max_new_tokens=340,
    )

    cleaned = re.sub(r"\s+", " ", answer or "").strip()
    cleaned = re.sub(r"^\s*model answer\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\d+\.\s*", " ", cleaned)
    cleaned = _normalize_line(cleaned)

    def _sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]

    if len(_sentences(cleaned)) < 5 or not cleaned.endswith((".", "!", "?")):
        continuation = _run_chat(
            system_prompt="You are a candidate in a technical interview.",
            user_prompt=(
                "Continue and finish this answer so it becomes 5 to 7 coherent complete sentences.\n"
                f"Current answer:\n{cleaned}"
            ),
            max_new_tokens=220,
        )
        cleaned = _normalize_line(f"{cleaned} {continuation}")

    sentences = _sentences(cleaned)
    if len(sentences) < 5:
        rewritten = _run_chat(
            system_prompt="You are a candidate in a technical interview.",
            user_prompt=(
                "Write a fresh direct answer as if you are the candidate speaking in an interview.\n"
                "Return 5 to 7 complete coherent sentences only.\n"
                f"Question: {question}"
            ),
            max_new_tokens=280,
        )
        cleaned = _normalize_line(rewritten)
        sentences = _sentences(cleaned)

    if len(sentences) > 7:
        sentences = sentences[:7]
    cleaned = _normalize_line(" ".join(sentences))
    return f"Model Answer: {cleaned}"


@lru_cache(maxsize=1024)
def generate_interviewer_reply(
    question: str,
    answer: str,
    role: str,
    tone: str,
    persona: str,
    score: int,
    strengths: str,
    improvements: str,
) -> str:
    tone_value = _normalize_tone(tone)
    persona_value = _normalize_persona(persona)
    safe_answer = (answer or "").strip() or "No candidate answer provided."
    safe_strengths = _normalize_line(strengths)
    safe_improvements = _normalize_line(improvements)
    safe_score = _coerce_score(score)

    prompt = (
        "You are the interviewer speaking directly to the candidate after hearing their answer.\n"
        "Write 2 to 3 natural conversational sentences.\n"
        "Rules:\n"
        "- sound human, not robotic\n"
        "- mention one specific positive from their answer\n"
        "- mention one concrete improvement\n"
        "- do not include headings, bullet points, JSON, or score formatting\n\n"
        f"Role: {role}\n"
        f"Interviewer tone: {TONE_LABELS[tone_value]}\n"
        f"Interviewer persona: {persona_value} ({PERSONA_INSTRUCTIONS[persona_value]})\n"
        f"Question: {question}\n"
        f"Candidate answer: {safe_answer}\n"
        f"Score context: {safe_score}/10\n"
        f"Grounded strengths: {safe_strengths}\n"
        f"Grounded improvements: {safe_improvements}\n"
    )

    try:
        text = _run_chat(
            system_prompt="You are a thoughtful technical interviewer giving spoken-style feedback.",
            user_prompt=prompt,
            max_new_tokens=140,
        )
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        cleaned = re.sub(r"^(interviewer|feedback)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        if cleaned:
            return cleaned
    except Exception:
        pass

    return (
        f"Thanks for your answer. {safe_strengths} "
        f"To level this up further, {safe_improvements[0].lower() + safe_improvements[1:] if safe_improvements else 'add more technical depth and one concrete example.'}"
    )


def _sanitize_question_line(text: str) -> str:
    line = re.sub(r"^\s*[-*\d\).:]+\s*", "", str(text or "").strip())
    line = re.sub(r"\s+", " ", line).strip()
    if not line:
        return ""
    if "?" not in line:
        line = line.rstrip(".") + "?"
    return line


def generate_custom_questions(
    role: str,
    skills: List[str],
    jd_context: str,
    count: int,
    persona: str = "Friendly Mentor",
) -> List[str]:
    if count <= 0:
        return []
    persona_value = _normalize_persona(persona)
    skill_text = ", ".join((skills or [])[:10]) or "general software engineering"
    prompt = (
        "Generate custom technical interview questions tailored to this candidate.\n"
        "Return strict JSON with one key: questions (array of strings).\n"
        f"Need exactly {count} questions.\n"
        "Each question must be concise, role-specific, and answerable in an interview.\n\n"
        f"Role: {role}\n"
        f"Interviewer persona: {persona_value} ({PERSONA_INSTRUCTIONS[persona_value]})\n"
        f"Candidate skills: {skill_text}\n"
        f"Job description context: {jd_context[:1200] or '[none]'}\n"
    )
    try:
        raw = _run_chat(
            system_prompt="You create high-signal interview question sets.",
            user_prompt=prompt,
            max_new_tokens=min(220, 70 + count * 35),
        )
        parsed = _extract_json_block(raw)
        items = parsed.get("questions") if isinstance(parsed, dict) else None
        out: List[str] = []
        if isinstance(items, list):
            for item in items:
                q = _sanitize_question_line(str(item))
                if q and q not in out:
                    out.append(q)
                if len(out) >= count:
                    break
        if out:
            return out[:count]
    except Exception:
        pass
    return []


def generate_followup_question(
    question: str,
    answer: str,
    role: str,
    score: int,
    tone: str = "medium",
    persona: str = "Friendly Mentor",
) -> Optional[Dict[str, str]]:
    score_i = _coerce_score(score)
    if 7 <= score_i <= 7:
        return None

    mode = "deepen" if score_i <= 6 else "stretch"
    tone_value = _normalize_tone(tone)
    persona_value = _normalize_persona(persona)
    mode_instruction = (
        "Ask one clarifying/deepening follow-up that helps the candidate strengthen missing details."
        if mode == "deepen"
        else "Ask one higher-difficulty stretch follow-up that tests deeper reasoning."
    )
    prompt = (
        f"{mode_instruction}\n"
        "Return only one follow-up interview question.\n"
        "Do not add explanation.\n\n"
        f"Role: {role}\n"
        f"Interviewer tone: {TONE_LABELS[tone_value]}\n"
        f"Interviewer persona: {persona_value} ({PERSONA_INSTRUCTIONS[persona_value]})\n"
        f"Original question: {question}\n"
        f"Candidate answer: {answer}\n"
        f"Score context: {score_i}/10\n"
    )
    try:
        raw = _run_chat(
            system_prompt="You are a technical interviewer generating one follow-up question.",
            user_prompt=prompt,
            max_new_tokens=90,
        )
        q = _sanitize_question_line(raw)
        if q:
            label = "FOLLOW-UP:DEEPEN" if mode == "deepen" else "FOLLOW-UP:STRETCH"
            return {"question": q, "source": label, "mode": mode}
    except Exception:
        pass

    if mode == "deepen":
        fallback = f"Can you walk me through a concrete implementation for your approach to: {question}"
        return {"question": _sanitize_question_line(fallback), "source": "FOLLOW-UP:DEEPEN", "mode": mode}
    fallback = f"What trade-offs would you make if scale or latency constraints became much tighter for: {question}"
    return {"question": _sanitize_question_line(fallback), "source": "FOLLOW-UP:STRETCH", "mode": mode}


def _heuristic_hiring_signal(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not evaluations:
        return {"decision": "NO_HIRE", "rationale": "Insufficient interview evidence.", "evidence": []}
    avg = sum(int(e.get("score", 1)) for e in evaluations) / len(evaluations)
    if avg >= 8.2:
        decision = "STRONG_HIRE"
    elif avg >= 6.4:
        decision = "HIRE"
    else:
        decision = "NO_HIRE"
    evidence: List[str] = []
    for ev in evaluations[:3]:
        s = _normalize_line(ev.get("strengths", ""))
        if s:
            evidence.append(s)
    return {
        "decision": decision,
        "rationale": f"Average interview performance was {avg:.2f}/10 across evaluated questions.",
        "evidence": evidence[:3],
    }


def summarize_hiring_signal(
    role: str,
    level: str,
    tone: str,
    persona: str,
    questions: List[str],
    answers: List[str],
    evaluations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    valid = []
    for i, ev in enumerate(evaluations):
        if ev is None:
            continue
        valid.append(
            {
                "question": questions[i] if i < len(questions) else "",
                "answer": answers[i] if i < len(answers) else "",
                "score": ev.get("score", 1),
                "strengths": ev.get("strengths", ""),
                "improvements": ev.get("improvements", ""),
            }
        )
    if not valid:
        return _heuristic_hiring_signal([])

    tone_value = _normalize_tone(tone)
    persona_value = _normalize_persona(persona)
    prompt = (
        "You are deciding an interview hiring signal.\n"
        "Return strict JSON keys: decision, rationale, evidence.\n"
        "- decision must be one of: STRONG_HIRE, HIRE, NO_HIRE\n"
        "- rationale: 1-2 sentences\n"
        "- evidence: array of 2-4 concise bullet-style strings grounded in candidate answers\n\n"
        f"Role: {role}\n"
        f"Level: {level}\n"
        f"Interviewer tone: {TONE_LABELS[tone_value]}\n"
        f"Interviewer persona: {persona_value} ({PERSONA_INSTRUCTIONS[persona_value]})\n"
        f"Interview data JSON:\n{json.dumps(valid)[:7000]}"
    )
    try:
        raw = _run_chat(
            system_prompt="You are an experienced hiring committee member.",
            user_prompt=prompt,
            max_new_tokens=240,
        )
        parsed = _extract_json_block(raw)
        if isinstance(parsed, dict):
            decision = str(parsed.get("decision", "NO_HIRE")).strip().upper()
            if decision not in {"STRONG_HIRE", "HIRE", "NO_HIRE"}:
                decision = "NO_HIRE"
            rationale = _normalize_line(parsed.get("rationale", ""))
            evidence_raw = parsed.get("evidence", [])
            evidence: List[str] = []
            if isinstance(evidence_raw, list):
                for item in evidence_raw:
                    line = _normalize_line(item)
                    if line:
                        evidence.append(line)
                    if len(evidence) >= 4:
                        break
            if not evidence:
                evidence = _heuristic_hiring_signal([v for v in valid]).get("evidence", [])
            return {"decision": decision, "rationale": rationale, "evidence": evidence[:4]}
    except Exception:
        pass
    return _heuristic_hiring_signal([v for v in valid])


def generate_debrief_report(
    role: str,
    level: str,
    tone: str,
    persona: str,
    questions: List[str],
    answers: List[str],
    evaluations: List[Dict[str, Any]],
) -> str:
    tone_value = _normalize_tone(tone)
    persona_value = _normalize_persona(persona)
    valid = []
    for i, ev in enumerate(evaluations):
        if ev is None:
            continue
        valid.append(
            {
                "question": questions[i] if i < len(questions) else "",
                "answer": answers[i] if i < len(answers) else "",
                "score": ev.get("score", 1),
                "strengths": ev.get("strengths", ""),
                "improvements": ev.get("improvements", ""),
            }
        )
    if not valid:
        return "# Interview Debrief\n\nNo completed answers yet."

    prompt = (
        "Write a concise markdown debrief report for the candidate.\n"
        "Include sections exactly:\n"
        "1) Overall Summary\n"
        "2) Top Strengths\n"
        "3) Gaps / Missed Concepts\n"
        "4) 7-Day Improvement Plan (daily bullets)\n"
        "5) Next Mock Interview Strategy\n\n"
        f"Role: {role}\n"
        f"Level: {level}\n"
        f"Interviewer tone: {TONE_LABELS[tone_value]}\n"
        f"Interviewer persona: {persona_value} ({PERSONA_INSTRUCTIONS[persona_value]})\n"
        f"Interview data JSON:\n{json.dumps(valid)[:7000]}"
    )
    try:
        text = _run_chat(
            system_prompt="You are a senior interview coach creating actionable prep plans.",
            user_prompt=prompt,
            max_new_tokens=520,
        )
        cleaned = str(text or "").strip()
        if cleaned:
            return cleaned
    except Exception:
        pass

    avg = sum(int(v.get("score", 1)) for v in valid) / len(valid)
    return (
        "# Interview Debrief\n\n"
        f"## Overall Summary\nCurrent average score: **{avg:.2f}/10** for **{role} ({level})**.\n\n"
        "## Top Strengths\n- You referenced relevant technical concepts from the asked questions.\n\n"
        "## Gaps / Missed Concepts\n- Add deeper implementation specifics and explicit trade-off discussion.\n\n"
        "## 7-Day Improvement Plan (daily bullets)\n"
        "- Day 1: Re-answer weakest two questions with concrete examples.\n"
        "- Day 2: Practice explaining one system trade-off per answer.\n"
        "- Day 3: Drill role-specific fundamentals for 60 minutes.\n"
        "- Day 4: Do a timed mock round (4 questions).\n"
        "- Day 5: Review and rewrite answers with measurable impact.\n"
        "- Day 6: Practice concise communication and structure.\n"
        "- Day 7: Full mock interview and retrospective.\n\n"
        "## Next Mock Interview Strategy\nFocus on depth first, then concision. Include one implementation example and one trade-off in every answer."
    )
