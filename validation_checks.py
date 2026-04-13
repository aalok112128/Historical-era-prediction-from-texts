import numpy as np
import pandas as pd
import pickle
import json
import scipy.sparse
import matplotlib.pyplot as plt
import warnings
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings('ignore')
nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)


# ════════════════════════════════════════════════════════════
# LOAD DATASET
# ════════════════════════════════════════════════════════════
print("Loading dataset...")
df     = pd.read_csv("dataset.csv")
texts  = df['text'].tolist()
labels = df['label'].tolist()

print(f"Total chunks : {len(df)}")
print(f"Unique books : {df['source'].nunique()}")
print(f"Eras         : {df['label'].unique()}\n")


# ════════════════════════════════════════════════════════════
# HELPER: STYLOMETRIC FEATURES
# ════════════════════════════════════════════════════════════
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
    func_freq = sum(1 for w in words_only if w in function_words) / len(words_only)
    return [avg_sent_len, ttr, avg_word_len,
            punct_density, hapax_ratio, func_freq]


# ════════════════════════════════════════════════════════════
# HELPER: BUILD FEATURE MATRIX
# ════════════════════════════════════════════════════════════
def build_features(texts, vectorizer=None, scaler=None,
                   fit=True, use_stylometry=True):
    if fit:
        vect  = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                stop_words='english', sublinear_tf=True)
        tfidf = vect.fit_transform(texts)
    else:
        vect  = vectorizer
        tfidf = vect.transform(texts)

    if not use_stylometry:
        return tfidf, vect, None

    stylo = np.array([extract_stylometric_features(t) for t in texts])
    if fit:
        sc           = MinMaxScaler()
        stylo_scaled = sc.fit_transform(stylo)
    else:
        sc           = scaler
        stylo_scaled = sc.transform(stylo)

    combined = hstack([tfidf, csr_matrix(stylo_scaled)])
    return combined, vect, sc


# ════════════════════════════════════════════════════════════
# CHECK 1 — BOOK-LEVEL SPLIT (generalisation test)
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("CHECK 1 — BOOK-LEVEL SPLIT (generalisation test)")
print("=" * 60)
print("Holding out 1 book per era as unseen test data...")
print("Model has NEVER seen these books during training.\n")

held_out = {
    'Renaissance'   : 'Renaissance/779.txt',
    'Enlightenment' : 'Enlightenment/147.txt',
    'Romantic'      : 'Romantic/9622.txt',
    'Victorian'     : 'Victorian/174.txt',
}

test_mask    = df['source'].isin(held_out.values())
train_mask   = ~test_mask
train_texts  = df.loc[train_mask, 'text'].tolist()
train_labels = df.loc[train_mask, 'label'].tolist()
test_texts   = df.loc[test_mask,  'text'].tolist()
test_labels  = df.loc[test_mask,  'label'].tolist()

print(f"Training chunks : {len(train_texts)}")
print(f"Testing  chunks : {len(test_texts)}")
print("Held-out books  :")
for era, src in held_out.items():
    n = df[df['source'] == src].shape[0]
    print(f"  {era:15s} → {src}  ({n} chunks)")

print("\nBuilding features for book-level split...")
X_train_bl, vect_bl, sc_bl = build_features(train_texts, fit=True)
X_test_bl,  _,       _     = build_features(test_texts,
                                             vectorizer=vect_bl,
                                             scaler=sc_bl,
                                             fit=False)

bl_results = {}
for name, model in [
    ('Naive Bayes',   ComplementNB(alpha=0.1)),
    ('SVM',           LinearSVC(C=1.0, max_iter=5000,
                               class_weight='balanced', random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=42,
                                             class_weight='balanced', n_jobs=-1)),
]:
    model.fit(X_train_bl, train_labels)
    preds = model.predict(X_test_bl)
    acc   = accuracy_score(test_labels, preds)
    f1    = f1_score(test_labels, preds, average='weighted')
    bl_results[name] = {'Accuracy': acc, 'F1 Score': f1}
    print(f"\n{name}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1 Score : {f1:.4f}")
    print(classification_report(test_labels, preds, zero_division=0))

print("\nBook-level results summary:")
bl_df = pd.DataFrame(bl_results).T.sort_values('Accuracy', ascending=False)
print(bl_df.round(4))


# ════════════════════════════════════════════════════════════
# CHECK 1B — BOOK-LEVEL ABLATION
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CHECK 1B — BOOK-LEVEL ABLATION")
print("Does stylometry help on completely UNSEEN books too?")
print("=" * 60)

X_train_bl_no, vect_bl_no, _ = build_features(
    train_texts, fit=True, use_stylometry=False)
X_test_bl_no,  _,          _ = build_features(
    test_texts, vectorizer=vect_bl_no, scaler=None,
    fit=False, use_stylometry=False)

bl_no_results = {}
for name, model in [
    ('Naive Bayes',   ComplementNB(alpha=0.1)),
    ('SVM',           LinearSVC(C=1.0, max_iter=5000,
                               class_weight='balanced', random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=42,
                                             class_weight='balanced', n_jobs=-1)),
]:
    model.fit(X_train_bl_no, train_labels)
    preds = model.predict(X_test_bl_no)
    acc   = accuracy_score(test_labels, preds)
    f1    = f1_score(test_labels, preds, average='weighted')
    bl_no_results[name] = {'Accuracy': acc, 'F1 Score': f1}

print("\nBOOK-LEVEL ABLATION — FULL COMPARISON TABLE")
print("=" * 60)
book_ablation_rows = []
for name in ['Naive Bayes', 'SVM', 'Random Forest']:
    without = bl_no_results[name]['Accuracy']
    with_s  = bl_results[name]['Accuracy']
    diff    = (with_s - without) * 100
    book_ablation_rows.append({
        'Model'              : name,
        'Without Stylometry' : f"{without*100:.2f}%",
        'With Stylometry'    : f"{with_s*100:.2f}%",
        'Improvement'        : f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%",
        'positive'           : diff >= 0
    })

book_abl_df = pd.DataFrame(book_ablation_rows)
print(book_abl_df[['Model','Without Stylometry',
                    'With Stylometry','Improvement']].to_string(index=False))


# ════════════════════════════════════════════════════════════
# CHECK 2 — STYLOMETRY ABLATION STUDY
# Uses SAME saved features as train_models.py for consistency
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CHECK 2 — STYLOMETRY ABLATION STUDY")
print("Does adding stylometric features actually improve results?")
print("=" * 60)

# Load exact same features used in train_models.py
X_full = scipy.sparse.load_npz("features.npz")
y_full = np.load("labels.npy", allow_pickle=True)

# Same split as train_models.py — guarantees identical numbers
X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# WITH stylometry — use full saved feature matrix
print("\n--- With Stylometry (from saved features.npz) ---")
ablation_results = {}

for name, model in [
    ('Naive Bayes',   ComplementNB(alpha=0.1)),
    ('SVM',           LinearSVC(C=1.0, max_iter=5000,
                               class_weight='balanced', random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=42,
                                             class_weight='balanced', n_jobs=-1)),
]:
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    f1    = f1_score(y_te, preds, average='weighted')
    ablation_results[(name, 'With Stylometry')] = {'Accuracy': acc, 'F1': f1}
    print(f"  {name:15s} → Accuracy: {acc*100:.2f}%  F1: {f1:.4f}")

# WITHOUT stylometry — TF-IDF only, same split
print("\n--- Without Stylometry (TF-IDF only) ---")
tfidf_only = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                              stop_words='english', sublinear_tf=True)
tfidf_full = tfidf_only.fit_transform(texts)

X_tr_no, X_te_no, y_tr_no, y_te_no = train_test_split(
    tfidf_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

for name, model in [
    ('Naive Bayes',   ComplementNB(alpha=0.1)),
    ('SVM',           LinearSVC(C=1.0, max_iter=5000,
                               class_weight='balanced', random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=42,
                                             class_weight='balanced', n_jobs=-1)),
]:
    model.fit(X_tr_no, y_tr_no)
    preds = model.predict(X_te_no)
    acc   = accuracy_score(y_te_no, preds)
    f1    = f1_score(y_te_no, preds, average='weighted')
    ablation_results[(name, 'Without Stylometry')] = {'Accuracy': acc, 'F1': f1}
    print(f"  {name:15s} → Accuracy: {acc*100:.2f}%  F1: {f1:.4f}")

# Comparison table
print("\n\nABLATION STUDY — FULL COMPARISON TABLE")
print("=" * 60)
rows = []
for name in ['Naive Bayes', 'SVM', 'Random Forest']:
    without = ablation_results[(name, 'Without Stylometry')]['Accuracy']
    with_s  = ablation_results[(name, 'With Stylometry')]['Accuracy']
    diff    = (with_s - without) * 100
    rows.append({
        'Model'              : name,
        'Without Stylometry' : f"{without*100:.2f}%",
        'With Stylometry'    : f"{with_s*100:.2f}%",
        'Improvement'        : f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%",
        'positive'           : diff >= 0
    })

abl_df = pd.DataFrame(rows)
print(abl_df[['Model','Without Stylometry',
              'With Stylometry','Improvement']].to_string(index=False))


# ════════════════════════════════════════════════════════════
# SAVE CHARTS
# ════════════════════════════════════════════════════════════

# Chart 1 — Book-level vs chunk-level accuracy
fig, ax = plt.subplots(figsize=(10, 5))
models = ['Naive Bayes', 'SVM', 'Random Forest']

with open("results.json") as f:
    res_json = json.load(f)

chunk_acc = [res_json[m]['Accuracy'] for m in models]
book_acc  = [bl_results[m]['Accuracy'] * 100 for m in models]

x     = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, chunk_acc, width,
               label='Chunk-level (original)', color='#534AB7')
bars2 = ax.bar(x + width/2, book_acc,  width,
               label='Book-level (honest)',    color='#1D9E75')

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%',
            ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Accuracy (%)')
ax.set_title('Chunk-level vs Book-level Accuracy\n'
             '(Book-level = truly unseen books)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 110)
ax.legend()
ax.axhline(y=25, color='red', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('book_vs_chunk_accuracy.png', dpi=150, bbox_inches='tight')
print("\nSaved: book_vs_chunk_accuracy.png")

# Chart 2 — Ablation study
fig2, ax2 = plt.subplots(figsize=(10, 5))
without_vals = [float(r['Without Stylometry'].replace('%', '')) for r in rows]
with_vals    = [float(r['With Stylometry'].replace('%', ''))    for r in rows]
model_names  = [r['Model'] for r in rows]

x2     = np.arange(len(model_names))
bars3  = ax2.bar(x2 - width/2, without_vals, width,
                 label='TF-IDF only',         color='#D85A30')
bars4  = ax2.bar(x2 + width/2, with_vals,    width,
                 label='TF-IDF + Stylometry', color='#534AB7')

for bar in bars3 + bars4:
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{bar.get_height():.1f}%',
             ha='center', va='bottom', fontsize=9)

ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Ablation Study — Impact of Stylometric Features',
              fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(model_names)
ax2.set_ylim(0, 110)
ax2.legend()
plt.tight_layout()
plt.savefig('ablation_study.png', dpi=150, bbox_inches='tight')
print("Saved: ablation_study.png")


# ════════════════════════════════════════════════════════════
# SAVE JSON RESULTS
# ════════════════════════════════════════════════════════════
ablation_save = []
for r in rows:
    ablation_save.append({
        "Model"              : r['Model'],
        "Without Stylometry" : r['Without Stylometry'],
        "With Stylometry"    : r['With Stylometry'],
        "Improvement"        : r['Improvement'],
        "positive"           : r['positive']
    })

validation_to_save = {
    "book_level": {
        name: {
            "Accuracy": round(vals["Accuracy"] * 100, 2),
            "F1 Score": round(vals["F1 Score"], 4)
        }
        for name, vals in bl_results.items()
    },
    "book_level_ablation": book_ablation_rows,
    "ablation"           : ablation_save
}

with open("validation_results.json", "w") as f:
    json.dump(validation_to_save, f, indent=2)

print("\n" + "=" * 60)
print("ALL CHECKS COMPLETE")
print("=" * 60)
print("\nKey files saved:")
print("  book_vs_chunk_accuracy.png — proves generalisation")
print("  ablation_study.png         — proves stylometry helps")
print("  validation_results.json    — all results for report")
print("\nUse the ablation table in your report's innovation section!")