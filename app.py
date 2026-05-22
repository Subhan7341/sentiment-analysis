# ================================================================
#  app.py  —  Sentiment Analysis Web App
#  Author : Muhammad Subhan Ahmad
#  GitHub : github.com/Subhan7341
# ================================================================
#
#  WHAT THIS FILE DOES  (say this in interviews):
#  ------------------------------------------------
#  This is the Flask backend. Flask is a Python web framework —
#  it listens for requests from the browser, runs our ML models,
#  and sends back the results as JSON.
#
#  The ML pipeline uses TWO models (ensemble approach):
#    1. TextBlob  — Naive Bayes classifier trained on movie reviews
#    2. VADER     — Rule-based lexicon, great for social media text
#
#  Averaging both scores gives more reliable results than
#  either model alone. This is called an "ensemble method."
# ================================================================

from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re

# ── Download NLTK data (runs once, safe to call every time) ──
# VADER needs a "lexicon" — a dictionary of ~7500 words with
# pre-assigned sentiment scores. nltk.download fetches it.
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt',         quiet=True)
nltk.download('punkt_tab',     quiet=True)

app = Flask(__name__)

# ── Initialize VADER once at startup, NOT inside the function ──
# INTERVIEW TIP: Loading a model is expensive (slow). If we put
# it inside the function, it reloads on EVERY request. Putting it
# here means it loads once and stays in memory. This is called
# "model caching" and is a standard production practice.
vader = SentimentIntensityAnalyzer()


# ────────────────────────────────────────────────────────────────
#  HELPER: clean the text before analysis
#  INTERVIEW TIP: This step is called "preprocessing" or
#  "data cleaning". Real-world text is messy — extra spaces,
#  weird characters etc. Cleaning it first improves accuracy.
# ────────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.strip()              # remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text) # collapse multiple spaces into one
    return text


# ────────────────────────────────────────────────────────────────
#  CORE: run both models and combine their scores
# ────────────────────────────────────────────────────────────────
def run_analysis(text):
    text = clean_text(text)

    # ── MODEL 1: TextBlob ────────────────────────────────────
    # polarity    : -1.0 (very negative) → 0 (neutral) → +1.0 (very positive)
    # subjectivity:  0.0 (objective/fact) → 1.0 (subjective/opinion)
    #
    # INTERVIEW TIP: TextBlob uses Naive Bayes — a probabilistic
    # algorithm based on Bayes Theorem. It assumes every word is
    # independent (the "naive" part). Trained on IMDb movie reviews.
    blob            = TextBlob(text)
    tb_polarity     = round(blob.sentiment.polarity,     4)
    tb_subjectivity = round(blob.sentiment.subjectivity, 4)

    # ── MODEL 2: VADER ───────────────────────────────────────
    # Returns: pos, neu, neg (each 0.0–1.0, they sum to 1.0)
    #          compound: overall score -1.0 → +1.0
    #
    # INTERVIEW TIP: VADER is rule-based, not ML. It understands
    # things like caps ("GREAT" > "great") and punctuation
    # ("great!!!" > "great"). Best for short social-media text.
    vs       = vader.polarity_scores(text)
    compound = round(vs['compound'], 4)

    # ── ENSEMBLE: average both scores ───────────────────────
    # INTERVIEW TIP: Using multiple models and combining results
    # is called an "ensemble method". It reduces each model's
    # individual weaknesses and improves overall accuracy.
    avg = round((tb_polarity + compound) / 2, 4)

    # ── FINAL LABEL ──────────────────────────────────────────
    # Standard VADER thresholds: >=0.05 positive, <=-0.05 negative
    if   avg >=  0.05: label, emoji, color = "Positive", "😊", "green"
    elif avg <= -0.05: label, emoji, color = "Negative", "😞", "red"
    else:              label, emoji, color = "Neutral",  "😐", "yellow"

    # Confidence = how far the score is from zero (0 = totally unsure)
    confidence = round(min(abs(avg) * 180, 99), 1)
    if confidence < 12:
        confidence = 12

    # ── TOKEN-LEVEL word scoring ─────────────────────────────
    # INTERVIEW TIP: In NLP a "token" is a single word.
    # We score each word individually to find which ones are
    # driving the sentiment — this is "token-level analysis".
    word_scores = []
    for w in text.split():
        clean_w = re.sub(r'[^a-zA-Z]', '', w).lower()
        if len(clean_w) > 2:
            s = vader.polarity_scores(clean_w)['compound']
            if abs(s) > 0.1:
                word_scores.append({
                    "word":      w,
                    "score":     round(s, 3),
                    "sentiment": "positive" if s > 0 else "negative"
                })
    word_scores.sort(key=lambda x: abs(x['score']), reverse=True)

    return {
        "label":      label,
        "emoji":      emoji,
        "color":      color,
        "confidence": confidence,
        "avg_score":  avg,
        "textblob": {
            "polarity":           tb_polarity,
            "subjectivity":       tb_subjectivity,
            "subjectivity_label": "Subjective" if tb_subjectivity > 0.5 else "Objective"
        },
        "vader": {
            "positive": round(vs['pos'] * 100, 1),
            "neutral":  round(vs['neu'] * 100, 1),
            "negative": round(vs['neg'] * 100, 1),
            "compound": compound
        },
        "key_words":  word_scores[:6],
        "word_count": len(text.split()),
        "char_count": len(text)
    }


# ────────────────────────────────────────────────────────────────
#  FLASK ROUTES
#
#  INTERVIEW TIP: A "route" maps a URL to a Python function.
#  When the browser visits that URL, Flask calls that function.
#
#  GET  /          → serve the HTML page
#  POST /analyze   → receive text JSON, return analysis JSON
#  GET  /examples  → return example sentences for the UI chips
# ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    # Always validate input before processing — defensive programming
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()

    if len(text) < 2:
        return jsonify({"error": "Text is too short to analyze"}), 400

    if len(text) > 5000:
        return jsonify({"error": "Max 5000 characters please"}), 400

    try:
        return jsonify(run_analysis(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/examples", methods=["GET"])
def examples():
    return jsonify([
        "I absolutely love this! It completely exceeded my expectations.",
        "This is the worst experience I have ever had. Totally disappointed.",
        "The package arrived on Tuesday. It was in a brown cardboard box.",
        "The movie was okay — some good parts but also quite slow.",
        "Brilliant work! The whole team performed exceptionally well.",
        "Terrible service. Cold food, rude staff. Never coming back.",
    ])


if __name__ == "__main__":
    # debug=True  → auto-restarts when you save a file change
    # port=5000   → open http://localhost:5000 in your browser
    app.run(debug=True, port=5000)
