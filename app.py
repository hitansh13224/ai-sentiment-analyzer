"""
AI Sentiment Analyzer Pro - Flask backend
Uses the Groq API (LLM-based) to detect emotion instead of keyword matching.
"""

import os
import json
import re
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_API_KEY = "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"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Minimum share of the normalized 100% a feeling must hold to be shown
# as "dominant" in the paragraph.
DOMINANCE_THRESHOLD = 5

# Every emotion bucketed into an overall tone. Used to roll the
# normalized per-emotion percentages up into Positive / Negative / Neutral.
POLARITY_MAP = {
    "Happy": "positive",
    "Excited": "positive",
    "Love": "positive",
    "Confident": "positive",
    "Hopeful": "positive",
    "Grateful": "positive",
    "Curious": "positive",
    "Sad": "negative",
    "Angry": "negative",
    "Nervous": "negative",
    "Fear": "negative",
    "Disgust": "negative",
    "Guilt": "negative",
    "Shame": "negative",
    "Jealous": "negative",
    "Lonely": "negative",
    "Frustrated": "negative",
    "Embarrassed": "negative",
    "Surprised": "neutral",
    "Neutral": "neutral",
    "Anxious": "negative",
    "Bored": "negative",
    "Calm": "positive",
    "Content": "positive",
    "Disappointed": "negative",
    "Enthusiastic": "positive",
    "Overwhelmed": "negative",
    "Proud": "positive",
    "Regretful": "negative",
    "Relieved": "positive",
    "Satisfied": "positive",
    "Suspicious": "negative",
    "Sympathetic": "neutral",
    "Anticipation": "neutral",
    "Determined": "positive",
}

POLARITY_EMOJI = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
POLARITY_LABEL = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}

# Fixed set of emotions the model must score (35 total).
EMOTIONS = [
    "Happy", "Sad", "Angry", "Nervous", "Fear", "Excited", "Love",
    "Surprised", "Confident", "Disgust", "Guilt", "Shame", "Jealous",
    "Hopeful", "Grateful", "Lonely", "Frustrated", "Embarrassed",
    "Curious", "Neutral",
    "Anxious", "Bored", "Calm", "Content", "Disappointed",
    "Enthusiastic", "Overwhelmed", "Proud", "Regretful", "Relieved",
    "Satisfied", "Suspicious", "Sympathetic", "Anticipation", "Determined"
]

EMOJI = {
    "Happy": "😊", "Sad": "😢", "Angry": "😡", "Nervous": "😰",
    "Fear": "😨", "Excited": "🤩", "Love": "❤️", "Surprised": "😲",
    "Confident": "😎", "Disgust": "🤢", "Guilt": "😔", "Shame": "😳",
    "Jealous": "😒", "Hopeful": "🤞", "Grateful": "🙏", "Lonely": "😞",
    "Frustrated": "😤", "Embarrassed": "😖", "Curious": "🤔",
    "Neutral": "😐",
    "Anxious": "😟", "Bored": "😑", "Calm": "😌", "Content": "🙂",
    "Disappointed": "😥", "Enthusiastic": "🙌", "Overwhelmed": "😵",
    "Proud": "🏆", "Regretful": "😣", "Relieved": "😅", "Satisfied": "👍",
    "Suspicious": "🤨", "Sympathetic": "🤗", "Anticipation": "👀",
    "Determined": "💪"
}

REPLY = {
    "Happy": "That's wonderful! 😊",
    "Sad": "Things will get better. 💙",
    "Angry": "Take a deep breath. 😌",
    "Nervous": "Believe in yourself. 🌸",
    "Fear": "You are stronger than you think.",
    "Excited": "Enjoy the moment! 🎉",
    "Love": "Wishing you happiness ❤️",
    "Surprised": "That sounds unexpected! 😲",
    "Confident": "Keep going! 💪",
    "Disgust": "That sounds unpleasant.",
    "Guilt": "Be kind to yourself.",
    "Shame": "You don't have to be hard on yourself.",
    "Jealous": "That's a completely human feeling.",
    "Hopeful": "Hold onto that hope! 🌟",
    "Grateful": "That's a beautiful thing to feel. 🙏",
    "Lonely": "You're not alone in this. 💙",
    "Frustrated": "Take a moment, you've got this.",
    "Embarrassed": "We've all been there!",
    "Curious": "Curiosity is a great sign! 🤔",
    "Neutral": "Thanks for sharing.",
    "Anxious": "Take it one step at a time. 🌿",
    "Bored": "Maybe it's time for something new!",
    "Calm": "That peaceful feeling is worth holding onto. 😌",
    "Content": "It's nice to feel settled. 🙂",
    "Disappointed": "It's okay to feel let down sometimes.",
    "Enthusiastic": "Love that energy! 🙌",
    "Overwhelmed": "One thing at a time, you've got this.",
    "Proud": "You've earned that feeling! 🏆",
    "Regretful": "Be gentle with yourself about the past.",
    "Relieved": "Glad that weight is off your shoulders. 😅",
    "Satisfied": "Nice to hear things are going well.",
    "Suspicious": "Trust your instincts, but stay open too.",
    "Sympathetic": "That's a kind way to feel.",
    "Anticipation": "Exciting things ahead! 👀",
    "Determined": "Keep pushing, you've got this. 💪"
}

FALLBACK_SCORES = {e: 0 for e in EMOTIONS}
FALLBACK_SCORES["Neutral"] = 100


# ---------------------------------------------------------------------------
# Groq call
# ---------------------------------------------------------------------------

def call_groq(text: str) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set on the server.")

    emotion_list = ", ".join(EMOTIONS)

    system_prompt = (
        "You are an emotion-analysis engine. Given a piece of text, score how "
        f"strongly each of the following emotions is present: {emotion_list}. "
        "Respond with ONLY a JSON object (no markdown, no commentary) matching "
        "exactly this shape:\n"
        "{\n"
        '  "scores": {"Happy": 0-100, "Sad": 0-100, ... every emotion listed above},\n'
        '  "keywords": ["word_or_phrase_from_text", ...]\n'
        "}\n"
        "Scores are integers 0-100 reflecting confidence that the text expresses "
        "that emotion. Most emotions should score low or 0; only emotions genuinely "
        "present should score high. If nothing stands out, Neutral should score high "
        "and the rest low. keywords should be short words or phrases copied from the "
        "input text that most influenced your scoring (max 8 items)."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "max_tokens": 1100,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    raw_content = resp.json()["choices"][0]["message"]["content"]

    # Defensive cleanup in case the model wraps the JSON in markdown fences.
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_content.strip(), flags=re.MULTILINE).strip()
    parsed = json.loads(cleaned)

    scores = parsed.get("scores", {})
    # Ensure every required emotion is present and clamp to 0-100 ints.
    full_scores = {}
    for e in EMOTIONS:
        try:
            val = int(round(float(scores.get(e, 0))))
        except (TypeError, ValueError):
            val = 0
        full_scores[e] = max(0, min(100, val))

    keywords = parsed.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k) for k in keywords][:8]

    return {"scores": full_scores, "keywords": keywords}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_scores(scores: dict) -> dict:
    """
    Rescale raw 0-100 emotion scores so that they sum to exactly 100,
    using the largest-remainder method so the result stays in whole
    numbers without losing/gaining a percentage point to rounding.
    """
    total = sum(scores.values())

    if total <= 0:
        # Nothing scored anything - treat as fully Neutral.
        result = {k: 0 for k in scores}
        if "Neutral" in result:
            result["Neutral"] = 100
        else:
            # Shouldn't happen given EMOTIONS always includes Neutral,
            # but guard anyway by dumping the remainder into the first key.
            first_key = next(iter(result), None)
            if first_key is not None:
                result[first_key] = 100
        return result

    exact = {k: (v / total) * 100 for k, v in scores.items()}
    floored = {k: int(exact[k]) for k in exact}
    remainder = 100 - sum(floored.values())

    # Give the leftover percentage points to whichever emotions had the
    # largest fractional remainder, so the total lands on exactly 100.
    by_fraction = sorted(
        exact.keys(),
        key=lambda k: exact[k] - floored[k],
        reverse=True,
    )

    idx = 0
    while remainder > 0 and by_fraction:
        key = by_fraction[idx % len(by_fraction)]
        floored[key] += 1
        remainder -= 1
        idx += 1

    return floored


def get_polarity_breakdown(normalized_scores: dict) -> dict:
    """
    Roll the normalized per-emotion percentages up into Positive / Negative /
    Neutral totals. Since normalized_scores already sums to 100, these three
    buckets will also sum to exactly 100.
    """
    totals = {"positive": 0, "negative": 0, "neutral": 0}

    for emotion, score in normalized_scores.items():
        bucket = POLARITY_MAP.get(emotion, "neutral")
        totals[bucket] += score

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    overall_label, overall_score = ranked[0]

    return {
        "positive": totals["positive"],
        "negative": totals["negative"],
        "neutral": totals["neutral"],
        "overall": POLARITY_LABEL[overall_label],
        "overall_emoji": POLARITY_EMOJI[overall_label],
    }


def get_dominant_emotions(normalized_scores: dict, threshold: int = DOMINANCE_THRESHOLD) -> list:
    """
    Return only the feelings whose normalized share exceeds the threshold,
    ranked from most to least dominant, each carrying its emoji.
    """
    ranked = sorted(normalized_scores.items(), key=lambda kv: kv[1], reverse=True)
    dominant = [
        {"emotion": name, "score": score, "emoji": EMOJI.get(name, "😐")}
        for name, score in ranked
        if score > threshold
    ]

    # Guarantee at least one entry (the top scorer) even if everything
    # is spread thin below the threshold.
    if not dominant and ranked:
        name, score = ranked[0]
        dominant = [{"emotion": name, "score": score, "emoji": EMOJI.get(name, "😐")}]

    return dominant


# ---------------------------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------------------------

def analyze_text(text: str) -> dict:
    try:
        groq_result = call_groq(text)
        raw_scores = groq_result["scores"]
        keywords = groq_result["keywords"]
    except Exception as exc:  # noqa: BLE001 - fall back gracefully
        raw_scores = dict(FALLBACK_SCORES)
        keywords = []
        raw_scores["_error"] = str(exc)

    error_msg = raw_scores.pop("_error", None)

    if not raw_scores or max(raw_scores.values()) == 0:
        raw_scores = dict(FALLBACK_SCORES)

    # Rescale so every response sums to exactly 100%.
    scores = normalize_scores(raw_scores)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    primary_emotion, primary_score = ranked[0]

    top_3 = [{"emotion": name, "score": score} for name, score in ranked[:3]]
    dominant_emotions = get_dominant_emotions(scores)
    polarity = get_polarity_breakdown(scores)

    if keywords:
        keyword_str = ", ".join(f'"{k}"' for k in keywords)
        explanation = (
            f"The AI identified {primary_emotion} as the dominant emotion, "
            f"influenced by words/phrases such as {keyword_str}."
        )
    else:
        explanation = f"The AI classified this text as primarily {primary_emotion}."

    result = {
        "sentiment": primary_emotion,
        "emoji": EMOJI.get(primary_emotion, "😐"),
        "confidence": primary_score,
        "top_3": top_3,
        "dominant_emotions": dominant_emotions,
        "dominance_threshold": DOMINANCE_THRESHOLD,
        "polarity": polarity,
        "explanation": explanation,
        "reply": REPLY.get(primary_emotion, "Thanks for sharing."),
        "keywords": keywords,
        "emotions": scores,
    }

    if error_msg:
        result["warning"] = f"Fell back to neutral analysis: {error_msg}"

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    if len(text) > 500:
        return jsonify({"error": "Text must be 500 characters or fewer."}), 400

    result = analyze_text(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)