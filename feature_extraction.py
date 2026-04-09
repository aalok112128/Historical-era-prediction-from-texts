import pandas as pd
import numpy as np
import pickle
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Load the dataset we created in preprocessing ────────────
print("Loading dataset...")
df = pd.read_csv("dataset.csv")
print(f"Loaded {len(df)} chunks across {df['label'].nunique()} eras")
print(df['label'].value_counts())

texts  = df['text'].tolist()
labels = df['label'].tolist()


# ── STEP 1: TF-IDF Features ─────────────────────────────────
# Converts each chunk into a vector of 5000 word-importance scores
print("\nExtracting TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,     # keep the 5000 most useful words
    ngram_range=(1, 2),    # single words AND 2-word pairs
    stop_words='english',  # remove common words like "the", "a"
    sublinear_tf=True      # apply log scaling — helps with long books
)
tfidf_matrix = vectorizer.fit_transform(texts)
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
# Expect: (3964, 5000)


# ── STEP 2: Stylometric Features ────────────────────────────
# 6 writing-style measurements per chunk — your innovation layer
stop_words_set = set(stopwords.words('english'))

def extract_stylometric_features(text):
    """Extract 6 style measurements from one text chunk."""
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('.')

    words_all  = word_tokenize(text.lower())
    words_only = [w for w in words_all if w.isalpha()]

    if len(words_only) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 1. Average sentence length
    avg_sent_len = len(words_only) / max(len(sentences), 1)

    # 2. Type-Token Ratio — vocabulary richness
    ttr = len(set(words_only)) / len(words_only)

    # 3. Average word length
    avg_word_len = sum(len(w) for w in words_only) / len(words_only)

    # 4. Punctuation density per 100 words
    punct_count   = sum(1 for ch in text if ch in '.,;:!?\'"-')
    punct_density = (punct_count / len(words_only)) * 100

    # 5. Hapax Legomena ratio — words appearing only once
    word_counts = Counter(words_only)
    hapax       = sum(1 for c in word_counts.values() if c == 1)
    hapax_ratio = hapax / len(set(words_only))

    # 6. Function word frequency
    function_words = {"the","of","and","a","in","to","is","was",
                      "it","that","he","she","his","her","they"}
    func_freq = sum(1 for w in words_only if w in function_words) / len(words_only)

    return [avg_sent_len, ttr, avg_word_len,
            punct_density, hapax_ratio, func_freq]

print("\nExtracting stylometric features (takes ~1-2 minutes)...")
stylo_list = []
for i, text in enumerate(texts):
    stylo_list.append(extract_stylometric_features(text))
    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{len(texts)} chunks...")

stylo_array  = np.array(stylo_list)
stylo_sparse = csr_matrix(stylo_array)
print(f"Stylometric matrix shape: {stylo_array.shape}")
# Expect: (3964, 6)


# ── STEP 3: Combine Both Feature Sets ───────────────────────
print("\nCombining TF-IDF + stylometric features...")
combined = hstack([tfidf_matrix, stylo_sparse])
print(f"Combined matrix shape: {combined.shape}")
# Expect: (3964, 5006)


# ── STEP 4: Save Everything ──────────────────────────────────
# Save features, labels, and vectorizer so train.py can load them
print("\nSaving features...")
import scipy.sparse
scipy.sparse.save_npz("features.npz", combined)

np.save("labels.npy", np.array(labels))

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("stylo_feature_names.txt", "w") as f:
    f.write("avg_sentence_length\n")
    f.write("type_token_ratio\n")
    f.write("avg_word_length\n")
    f.write("punct_density\n")
    f.write("hapax_ratio\n")
    f.write("function_word_freq\n")

print("\n" + "="*50)
print("FEATURE EXTRACTION SUMMARY")
print("="*50)
print(f"Total samples      : {combined.shape[0]}")
print(f"TF-IDF features    : 5000")
print(f"Stylometric features: 6")
print(f"Total features     : {combined.shape[1]}")
print("\nFiles saved:")
print("  features.npz    — combined feature matrix")
print("  labels.npy      — era labels")
print("  vectorizer.pkl  — TF-IDF vectorizer (needed for web demo)")
print("\nYou are ready for model training!")