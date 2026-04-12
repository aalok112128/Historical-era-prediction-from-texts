from flask import Flask, request, jsonify, render_template
import pickle
import scipy.sparse
import numpy as np
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import os

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

# ── Load saved model and vectorizer ─────────────────────────
# Go one level up from webapp/ to find the pkl files
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
VECT_PATH  = os.path.join(BASE_DIR, "vectorizer.pkl")

print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Loading vectorizer...")
with open(VECT_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("Model and vectorizer loaded successfully!")

# ── Era information for display ──────────────────────────────
ERA_INFO = {
    "Renaissance": {
        "years"   : "1500 – 1660",
        "authors" : "Shakespeare, Marlowe, Francis Bacon",
        "traits"  : "Archaic vocabulary, thee/thou/doth, short dramatic sentences",
        "color"   : "#534AB7"
    },
    "Enlightenment": {
        "years"   : "1660 – 1800",
        "authors" : "Swift, Defoe, Adam Smith",
        "traits"  : "Formal tone, Latin-derived words, rational argumentation",
        "color"   : "#0F6E56"
    },
    "Romantic": {
        "years"   : "1800 – 1850",
        "authors" : "Austen, Shelley, Wordsworth, Scott",
        "traits"  : "Emotional language, nature imagery, rich adjectives",
        "color"   : "#993C1D"
    },
    "Victorian": {
        "years"   : "1850 – 1920",
        "authors" : "Dickens, Wilde, Brontë, Joyce",
        "traits"  : "Long complex sentences, social commentary, moral themes",
        "color"   : "#185FA5"
    }
}

# ── Stylometric feature extractor ───────────────────────────
def extract_stylometric_features(text):
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('.')
    words_all  = word_tokenize(text.lower())
    words_only = [w for w in words_all if w.isalpha()]
    if len(words_only) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    avg_sent_len  = len(words_only) / max(len(sentences), 1)
    ttr           = len(set(words_only)) / len(words_only)
    avg_word_len  = sum(len(w) for w in words_only) / len(words_only)
    punct_count   = sum(1 for ch in text if ch in '.,;:!?\'"-')
    punct_density = (punct_count / len(words_only)) * 100
    word_counts   = Counter(words_only)
    hapax         = sum(1 for c in word_counts.values() if c == 1)
    hapax_ratio   = hapax / len(set(words_only))
    function_words = {"the","of","and","a","in","to","is","was",
                      "it","that","he","she","his","her","they"}
    func_freq = sum(1 for w in words_only if w in function_words)/len(words_only)
    return [avg_sent_len, ttr, avg_word_len,
            punct_density, hapax_ratio, func_freq]


def predict_era(text):
    """Full pipeline: text → features → prediction."""
    # Step 1 — TF-IDF
    tfidf = vectorizer.transform([text])

    # Step 2 — Stylometry
    stylo        = np.array([extract_stylometric_features(text)])
    stylo_sparse = csr_matrix(stylo)

    # Step 3 — Combine
    combined = hstack([tfidf, stylo_sparse])

    # Step 4 — Predict
    prediction = model.predict(combined)[0]

    # Step 5 — Confidence scores
    # LinearSVC doesn't have predict_proba so we use decision function
    try:
        proba  = model.predict_proba(combined)[0]
        labels = model.classes_
        confidence = {label: round(float(prob)*100, 1)
                      for label, prob in zip(labels, proba)}
    except AttributeError:
        # LinearSVC — use decision function scores instead
        scores     = model.decision_function(combined)[0]
        # Convert to positive values and normalise to percentages
        scores     = scores - scores.min()
        total      = scores.sum()
        labels     = model.classes_
        confidence = {label: round(float(score/total)*100, 1)
                      for label, score in zip(labels, scores)}

    # Step 6 — Stylometric signals for display
    feats    = extract_stylometric_features(text)
    signals  = {
        "Avg sentence length" : f"{feats[0]:.1f} words",
        "Vocabulary richness" : f"{feats[1]*100:.1f}%",
        "Avg word length"     : f"{feats[2]:.1f} chars",
        "Punctuation density" : f"{feats[3]:.1f} per 100 words",
        "Hapax ratio"         : f"{feats[4]*100:.1f}%",
        "Function word freq"  : f"{feats[5]*100:.1f}%",
    }

    return {
        "era"        : prediction,
        "info"       : ERA_INFO[prediction],
        "confidence" : confidence,
        "signals"    : signals,
        "word_count" : len(text.split())
    }


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if len(text.split()) < 50:
        return jsonify({"error": "Please enter at least 50 words for accurate prediction."}), 400

    result = predict_era(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)