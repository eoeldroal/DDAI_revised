import json
import math
import os
import re
from typing import Any, Iterable, Optional


LLM_JUDGE_PROMPT = """You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- the query
- a generated answer
- a reference answer

Your task is to evaluate the correctness of the generated answer.

## Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}

## Evaluation Guidelines
- Evaluate if the generated answer is semantically equivalent to the reference answer
- Consider semantic equivalence, not just exact string match
- If the core meaning is the same, judge it as correct even if the wording differs
- Be lenient with minor formatting differences (e.g., "$4.5B" vs "4.5 billion dollars")

## Scoring
Provide a score from 0.0 to 1.0:
- 1.0: Semantically equivalent (correct)
- 0.5-0.9: Partially correct
- 0.0: Incorrect or unrelated
"""

LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": "Semantic correctness score from 0.0 (wrong) to 1.0 (correct)"
        }
    },
    "required": ["score"]
}

_GEMINI_CLIENT = None

def _parse_env_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None, None
    if "=" not in line:
        return None, None
    key, val = line.split("=", 1)
    key = key.strip()
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        val = val[1:-1]
    return key, val


def _load_env_fallback() -> None:
    keys_to_load = {"GEMINI_API_KEY", "GEMINI_MODEL", "GOOGLE_CLOUD_PROJECT"}
    if all(os.environ.get(k) for k in keys_to_load):
        return

    candidates = []
    project_dir = os.environ.get("PROJECT_DIR")
    if project_dir:
        candidates.append(os.path.join(project_dir, ".env"))
    candidates.append(os.path.join(os.getcwd(), ".env"))

    try:
        cur = os.path.abspath(os.path.dirname(__file__))
        for _ in range(6):
            candidates.append(os.path.join(cur, ".env"))
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    except Exception:
        pass

    for env_path in candidates:
        if not env_path or not os.path.isfile(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    key, val = _parse_env_line(line)
                    if not key or val is None:
                        continue
                    if key in keys_to_load and not os.environ.get(key):
                        os.environ[key] = val
        except Exception:
            continue
        break


def _get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _load_env_fallback()
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai
    except Exception:
        return None

    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": LLM_RESPONSE_SCHEMA,
    }
    _GEMINI_CLIENT = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )
    return _GEMINI_CLIENT

def _parse_llm_score(text: str) -> float:
    try:
        result = json.loads(text)
        raw_score = result.get("score", 0.0)
        score = max(0.0, min(1.0, float(raw_score)))
        return score
    except Exception:
        return 0.0

def _extract_answer_from_response(text: str) -> Optional[str]:
    end_tag = "</answer>"
    start_tag = "<answer>"
    end_pos = text.rfind(end_tag)
    if end_pos == -1:
        return None
    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None
    start_pos += len(start_tag)
    return text[start_pos:end_pos]


def simple_format_checker(data_source, solution_str, ground_truth, extra_info):
    """
    Check assistant response format across turns.

    Returns:
        score (float): 1.0 (pass) or 0.0 (fail)
        reason (str | None): failure reason if any
    """
    assistant_turns = re.findall(r"<\|im_start\|>assistant(.*?)<\|im_end\|>", solution_str, re.DOTALL)

    if not assistant_turns:
        # Fallback: split by think-action pairs as turns
        pair_pattern = re.compile(r"(<think>.*?</think>\s*(?:<search>.*?</search>|<bbox>.*?</bbox>|<search_complete>true</search_complete>))", re.DOTALL)
        assistant_turns = pair_pattern.findall(solution_str)

    if not assistant_turns:
        assistant_turns = [solution_str]

    action_turns = []
    answer_only_turns = []

    for i, turn in enumerate(assistant_turns):
        cleaned_turn = turn.strip()
        action_count = (
            cleaned_turn.count("<search>")
            + cleaned_turn.count("<bbox>")
            + cleaned_turn.count("<search_complete>")
        )
        has_answer = ("<answer>" in cleaned_turn) or ("</answer>" in cleaned_turn)

        if action_count == 0 and has_answer:
            if cleaned_turn.startswith("<answer>") and cleaned_turn.endswith("</answer>"):
                answer_only_turns.append((i, cleaned_turn))
                continue
            return 0.0, f"Turn {i} contains malformed/mixed <answer> block"

        if action_count == 0:
            return 0.0, f"Turn {i} missing action tag"

        if has_answer:
            return 0.0, f"Turn {i} contains <answer> inside action turn"

        action_turns.append((i, cleaned_turn))

    if not action_turns:
        return 0.0, "No action turns found"

    for i, cleaned_turn in action_turns:
        # Enforce exactly one think block per turn
        if cleaned_turn.count("<think>") != 1 or cleaned_turn.count("</think>") != 1:
            return 0.0, f"Turn {i} incorrect <think> tag count. {cleaned_turn[:50]}..."

        if not cleaned_turn.startswith("<think>"):
            return 0.0, f"Turn {i} missing <think> start tag. {cleaned_turn[:50]}..."

        action_count = (
            cleaned_turn.count("<search>")
            + cleaned_turn.count("<bbox>")
            + cleaned_turn.count("<search_complete>")
        )
        if action_count != 1:
            return 0.0, f"Turn {i} invalid action count ({action_count})"

        if "<search>" in cleaned_turn:
            match = re.search(r"<search>(.*?)</search>", cleaned_turn, re.DOTALL)
            if not match or not match.group(1).strip():
                return 0.0, f"Turn {i} empty/malformed <search>"

        elif "<bbox>" in cleaned_turn:
            match = re.search(r"<bbox>(.*?)</bbox>", cleaned_turn, re.DOTALL)
            if not match:
                return 0.0, f"Turn {i} malformed <bbox>"
            try:
                bbox_content = json.loads(match.group(1).strip())
                if not isinstance(bbox_content, list) or len(bbox_content) != 4:
                    return 0.0, f"Turn {i} bbox format error (not length 4)"
                if not all(isinstance(coord, (int, float)) for coord in bbox_content):
                    return 0.0, f"Turn {i} bbox non-number values"
            except json.JSONDecodeError:
                return 0.0, f"Turn {i} bbox JSON decode error"

        elif "<search_complete>" in cleaned_turn:
            if "<search_complete>true</search_complete>" not in cleaned_turn.replace(" ", ""):
                return 0.0, f"Turn {i} <search_complete> value error"

    # Last action turn must be search_complete
    last_action_idx, last_action_turn = action_turns[-1]
    if "<search_complete>" not in last_action_turn:
        return 0.0, "Last action turn missing <search_complete>"

    # Answer-only turns allowed only after last action
    for idx, _turn in answer_only_turns:
        if idx < last_action_idx:
            return 0.0, "Answer block appears before search_complete"

    return 1.0, None


def _dcg(relevance_scores: Iterable[int]) -> float:
    dcg_value = 0.0
    for i, relevance in enumerate(relevance_scores, start=1):
        dcg_value += (2**relevance - 1) / math.log2(i + 1)
    return dcg_value


def _ndcg(sorted_docs: list[str], golden_answer_list: list[str]) -> float:
    relevance_scores = [1 if doc in golden_answer_list else 0 for doc in sorted_docs]
    dcg_value = _dcg(relevance_scores)

    ideal_relevance_scores = [1] * len(golden_answer_list) + [0] * (
        max(len(sorted_docs) - len(golden_answer_list), 0)
    )
    idcg_value = _dcg(ideal_relevance_scores)
    if idcg_value == 0:
        return 0.0
    return dcg_value / idcg_value


def _to_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_doc_id(doc: str) -> str:
    """Normalize doc id like '14_7' -> '7' when suffix is numeric."""
    s = str(doc)
    s = os.path.splitext(s)[0]
    if '_' in s:
        last = s.split('_')[-1]
        if last.isdigit():
            return last
    return s

def _basename_no_ext(path: Any) -> str:
    p = str(path).rstrip("/")
    base = os.path.basename(p)
    if base.endswith(".jpg"):
        base = base[:-4]
    return base


def _extract_retrieved(extra_info: dict) -> list[str]:
    for key in (
        "retrievaled_images",
        "retrieved_images",
        "retrievaled_image_paths",
        "retrieved_image_paths",
        "image_paths",
        "retrieved_documents",
    ):
        if key in extra_info:
            return _to_list(extra_info.get(key))
    return []


def _extract_reference(extra_info: dict) -> list[str]:
    reference_docs = extra_info.get("reference_documents")
    if reference_docs is None:
        reference_docs = extra_info.get("reference_page") or extra_info.get("reference_pages") or extra_info.get("pages")
    if reference_docs is not None:
        return [str(x) for x in _to_list(reference_docs)]

    file_name = extra_info.get("file_name") or extra_info.get("file") or extra_info.get("pdf_name")
    if not file_name:
        return []
    base = os.path.basename(str(file_name))
    if ".pdf" in base:
        stem = base.split(".pdf")[0]
    else:
        stem = os.path.splitext(base)[0]

    pages = extra_info.get("reference_page") or extra_info.get("reference_pages") or extra_info.get("pages")
    pages = _to_list(pages)
    return [f"{stem}_{page}" for page in pages]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    extra_info = extra_info or {}

    format_score, fail_reason = simple_format_checker(data_source, solution_str, ground_truth, extra_info)

    retrieved = _extract_retrieved(extra_info)
    retrieved_basenames = [_basename_no_ext(item) for item in retrieved]
    retrieved_norm = [_normalize_doc_id(x) for x in retrieved_basenames]

    reference_raw = _extract_reference(extra_info)
    reference_norm = [_normalize_doc_id(x) for x in reference_raw]

    ndcg_value = _ndcg(retrieved_norm, reference_norm)

    query = extra_info.get("question", "")
    generated_answer = _extract_answer_from_response(solution_str)
    if generated_answer is None:
        generated_answer = ""

    judge_score = 0.0
    judge_error = None
    judge_parse_ok = None
    judge_response_trunc = None
    client = _get_gemini_client()
    judge_model = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
    judge_api_key_set = bool(os.environ.get("GEMINI_API_KEY"))
    if client is None:
        judge_error = "no_client"
    else:
        prompt = LLM_JUDGE_PROMPT.format(
            query=query,
            generated_answer=generated_answer,
            reference_answer=ground_truth,
        )
        try:
            response = client.generate_content(prompt)
            response_text = getattr(response, "text", "") or ""
            judge_response_trunc = response_text[:200]
            try:
                json.loads(response_text)
                judge_parse_ok = True
            except Exception:
                judge_parse_ok = False
            judge_score = _parse_llm_score(response_text)
        except Exception as exc:
            judge_error = type(exc).__name__
            judge_score = 0.0

    try:
        judge_weight = float(os.environ.get("JUDGE_WEIGHT", "0.8"))
    except Exception:
        judge_weight = 0.8
    try:
        ndcg_weight = float(os.environ.get("NDCG_WEIGHT", "0.2"))
    except Exception:
        ndcg_weight = 0.2

    final_score = judge_weight * float(judge_score) + ndcg_weight * float(ndcg_value)

    return {
        "score": float(final_score),
        "judge_score": float(judge_score),
        "judge_weight": float(judge_weight),
        "ndcg_weight": float(ndcg_weight),
        "judge_error": judge_error,
        "judge_model": judge_model,
        "judge_api_key_set": judge_api_key_set,
        "judge_parse_ok": judge_parse_ok,
        "judge_response_trunc": judge_response_trunc,
        "format_score": float(format_score),
        "ndcg": float(ndcg_value),
        "format_fail_reason": fail_reason,
        # JSON strings to keep reward_extra_info homogeneous
        "retrieved_basenames_json": json.dumps(retrieved_basenames, ensure_ascii=False),
        "reference_docs_norm_json": json.dumps(reference_norm, ensure_ascii=False),
        "retrieved_count": len(retrieved_norm),
        "reference_count": len(reference_norm),
    }
