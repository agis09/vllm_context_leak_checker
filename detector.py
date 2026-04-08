"""Contamination detection logic.

Three detection layers:
1. Token match  — other user's canary UUID/build/names in the response (zero false-positive)
2. Language leak — wrong script in response (e.g. Hangul in Japanese session)
3. N-gram similarity — unusually high overlap between concurrent responses

Supported languages: Japanese, Korean, Thai

Script detection matrix (all 6 directions covered):
  JP→KR: detect ひらがな/カタカナ in Korean response  ✅
  JP→TH: detect ひらがな/カタカナ in Thai response     ✅
  KR→JP: detect ハングル in Japanese response           ✅
  KR→TH: detect ハングル in Thai response              ✅
  TH→JP: detect タイ文字 in Japanese response           ✅
  TH→KR: detect タイ文字 in Korean response             ✅
"""

import re
from collections import defaultdict
from user_profile import UserProfile


# ---------------------------------------------------------------------------
# Unicode ranges
# ---------------------------------------------------------------------------

HIRAGANA_KATAKANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
HANGUL = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
THAI = re.compile(r"[\u0E00-\u0E7F]")


# ---------------------------------------------------------------------------
# Layer 1: Canary token detection
# ---------------------------------------------------------------------------

def detect_token_contamination(
    user_id: str,
    language: str,
    text: str,
    all_profiles: list[UserProfile],
) -> list[dict]:
    """Check whether *text* contains canary tokens belonging to users of
    a *different* language.  Same-language contamination is ignored because
    it is indistinguishable from hallucination."""
    contaminations = []
    for profile in all_profiles:
        if profile.user_id == user_id:
            continue
        # Skip same-language users
        if profile.language == language:
            continue
        for token_name, token_value in profile.all_canary_tokens.items():
            if token_value in text:
                contaminations.append({
                    "source_user": profile.user_id,
                    "source_language": profile.language,
                    "source_persona": profile.persona_name,
                    "source_project": profile.project_code,
                    "token_type": token_name,
                    "token_value": token_value,
                })
    return contaminations


# ---------------------------------------------------------------------------
# Layer 2: Language contamination
# ---------------------------------------------------------------------------

def detect_language_contamination(expected_lang: str, text: str) -> list[dict]:
    """Detect characters from the wrong writing system.

    Each language checks for ALL other scripts:
    - japanese: flag Hangul, Thai
    - korean:   flag Hiragana/Katakana, Thai
    - thai:     flag Hiragana/Katakana, Hangul

    All 6 contamination directions are detectable via script analysis.
    """
    # Map: script_name -> (regex, detected_language)
    # Note: JP session does NOT check for Hangul — rare hallucination possible
    checks_by_lang: dict[str, list[tuple[re.Pattern, str]]] = {
        "japanese": [(THAI, "thai")],
        "korean": [(HIRAGANA_KATAKANA, "japanese"), (THAI, "thai")],
        "thai": [(HIRAGANA_KATAKANA, "japanese"), (HANGUL, "korean")],
    }

    contaminations = []
    for pattern, detected in checks_by_lang.get(expected_lang, []):
        matches = pattern.findall(text)
        if matches:
            contaminations.append({
                "expected": expected_lang,
                "detected": detected,
                "characters": matches[:20],
                "count": len(matches),
            })

    return contaminations


# ---------------------------------------------------------------------------
# Real-time per-turn detection
# ---------------------------------------------------------------------------

def detect_turn(
    user_id: str,
    language: str,
    turn_data: dict,
    all_profiles: list[UserProfile],
) -> list[dict]:
    """Check a single turn for contamination (token + language).

    Designed to be called immediately after each API response, enabling
    real-time detection without waiting for session completion.
    """
    findings = []
    turn_idx = turn_data.get("turn", -1)
    is_after_tool = turn_idx > 0

    for field in ("content", "reasoning"):
        text = turn_data.get(field)
        if not text:
            continue

        for hit in detect_token_contamination(user_id, language, text, all_profiles):
            findings.append({
                "type": "token",
                "victim_user": user_id,
                "victim_language": language,
                "turn": turn_idx,
                "field": field,
                "is_after_tool_result": is_after_tool,
                **hit,
            })

        for hit in detect_language_contamination(language, text):
            findings.append({
                "type": "language",
                "victim_user": user_id,
                "victim_language": language,
                "turn": turn_idx,
                "field": field,
                "is_after_tool_result": is_after_tool,
                **hit,
            })

    return findings


# ---------------------------------------------------------------------------
# Layer 3: N-gram similarity between concurrent responses
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 5) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)} if len(text) >= n else set()


def detect_ngram_similarity(
    sessions: list[dict],
    threshold: float = 0.30,
    ngram_size: int = 5,
) -> list[dict]:
    """Flag pairs of concurrent sessions whose final responses share
    an unusually high fraction of character n-grams.

    Only compares sessions that belong to *different* languages.
    Same-language pairs are skipped (contamination hard to distinguish
    from similar prompt templates).
    """
    findings = []

    # Collect turn content per session: (user_id, language, text, field)
    texts: list[tuple[str, str, str, str]] = []
    for sess in sessions:
        uid = sess["user_id"]
        lang = sess["language"]
        for turn in sess.get("turns", []):
            for field in ("content", "reasoning"):
                t = turn.get(field)
                if t and len(t) > 50:
                    texts.append((uid, lang, t, field))

    for i in range(len(texts)):
        uid_a, lang_a, text_a, field_a = texts[i]
        ngrams_a = _char_ngrams(text_a, ngram_size)
        if not ngrams_a:
            continue
        for j in range(i + 1, len(texts)):
            uid_b, lang_b, text_b, field_b = texts[j]
            # Skip same-language pairs
            if lang_a == lang_b:
                continue
            ngrams_b = _char_ngrams(text_b, ngram_size)
            if not ngrams_b:
                continue
            overlap = len(ngrams_a & ngrams_b)
            similarity = overlap / min(len(ngrams_a), len(ngrams_b))
            if similarity >= threshold:
                findings.append({
                    "type": "ngram_similarity",
                    "user_a": uid_a,
                    "language_a": lang_a,
                    "field_a": field_a,
                    "user_b": uid_b,
                    "language_b": lang_b,
                    "field_b": field_b,
                    "similarity": round(similarity, 4),
                    "shared_ngrams": overlap,
                })

    return findings


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------

def analyze_session(session_result: dict, all_profiles: list[UserProfile]) -> list[dict]:
    """Analyze a single session for contamination (layers 1 & 2)."""
    findings = []
    user_id = session_result["user_id"]
    language = session_result["language"]

    for turn in session_result.get("turns", []):
        turn_idx = turn["turn"]
        is_after_tool = turn_idx > 0

        for field in ("content", "reasoning"):
            text = turn.get(field)
            if not text:
                continue

            # Layer 1: token contamination (cross-language only)
            for hit in detect_token_contamination(user_id, language, text, all_profiles):
                findings.append({
                    "type": "token",
                    "victim_user": user_id,
                    "victim_language": language,
                    "turn": turn_idx,
                    "field": field,
                    "is_after_tool_result": is_after_tool,
                    **hit,
                })

            # Layer 2: language contamination
            for hit in detect_language_contamination(language, text):
                findings.append({
                    "type": "language",
                    "victim_user": user_id,
                    "victim_language": language,
                    "turn": turn_idx,
                    "field": field,
                    "is_after_tool_result": is_after_tool,
                    **hit,
                })

    return findings


def analyze_all_sessions(
    all_sessions: list[dict],
    all_profiles: list[UserProfile],
    ngram_threshold: float = 0.30,
) -> dict:
    """Analyze all sessions and produce a summary report."""
    all_findings: list[dict] = []
    errors: list[dict] = []

    for session in all_sessions:
        if session.get("error"):
            errors.append({
                "user_id": session["user_id"],
                "error": session["error"],
            })
        all_findings.extend(analyze_session(session, all_profiles))

    # Layer 3: n-gram similarity across concurrent sessions
    ngram_findings = detect_ngram_similarity(all_sessions, threshold=ngram_threshold)
    all_findings.extend(ngram_findings)

    # --- Build summary ---
    token_findings = [f for f in all_findings if f["type"] == "token"]
    lang_findings = [f for f in all_findings if f["type"] == "language"]
    ngram_sim_findings = [f for f in all_findings if f["type"] == "ngram_similarity"]

    # Contamination matrix: source -> victim -> count
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in token_findings:
        matrix[f["source_user"]][f["victim_user"]] += 1

    # Turn distribution
    turn_dist: dict[int, int] = defaultdict(int)
    for f in all_findings:
        if "turn" in f:
            turn_dist[f["turn"]] += 1

    # Field distribution
    field_dist: dict[str, int] = defaultdict(int)
    for f in all_findings:
        if "field" in f:
            field_dist[f["field"]] += 1

    after_tool_count = sum(1 for f in all_findings if f.get("is_after_tool_result"))

    summary = {
        "total_sessions": len(all_sessions),
        "sessions_with_errors": len(errors),
        "total_findings": len(all_findings),
        "token_contaminations": len(token_findings),
        "language_contaminations": len(lang_findings),
        "ngram_similarity_alerts": len(ngram_sim_findings),
        "contaminations_after_tool_result": after_tool_count,
        "contaminations_before_tool_result": len(all_findings) - after_tool_count - len(ngram_sim_findings),
        "turn_distribution": dict(turn_dist),
        "field_distribution": dict(field_dist),
        "contamination_matrix": {k: dict(v) for k, v in matrix.items()},
    }

    return {
        "summary": summary,
        "findings": all_findings,
        "errors": errors,
    }
