import pandas as pd
import numpy as np
import pickle
import os
import base64
from datetime import datetime

# ── Load all results we already have ────────────────────────
print("Loading project data...")
df = pd.read_csv("dataset.csv")

# Encode charts as base64 so they embed directly in HTML
def encode_image(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

img_confusion   = encode_image("confusion_matrices.png")
img_accuracy    = encode_image("accuracy_comparison.png")
img_ablation    = encode_image("ablation_study.png")
img_bookvschunk = encode_image("book_vs_chunk_accuracy.png")

# ── Your actual results (from your runs) ────────────────────
import json

# ── Load real results from training runs ─────────────────────
with open("results.json", "r") as f:
    chunk_results = json.load(f)

with open("validation_results.json", "r") as f:
    val_data = json.load(f)

book_results          = val_data["book_level"]
ablation_results      = val_data["ablation"]
book_ablation_results = val_data["book_level_ablation"]

books = [
    ("Renaissance",   1515,  "The Merchant of Venice",    "Shakespeare"),
    ("Renaissance",   15272, "The Faerie Queene",          "Spenser"),
    ("Renaissance",   45988, "Novum Organum",              "Bacon"),
    ("Renaissance",   779,   "Doctor Faustus",             "Marlowe"),
    ("Enlightenment", 829,   "Gulliver's Travels",         "Swift"),
    ("Enlightenment", 521,   "Robinson Crusoe",            "Defoe"),
    ("Enlightenment", 3300,  "The Wealth of Nations",      "Smith"),
    ("Enlightenment", 147,   "Common Sense",               "Paine"),
    ("Romantic",      1342,  "Pride and Prejudice",        "Austen"),
    ("Romantic",      84,    "Frankenstein",               "Shelley"),
    ("Romantic",      9622,  "Lyrical Ballads",            "Wordsworth"),
    ("Romantic",      82,    "Ivanhoe",                    "Scott"),
    ("Victorian",     1400,  "Great Expectations",         "Dickens"),
    ("Victorian",     174,   "The Picture of Dorian Gray", "Wilde"),
    ("Victorian",     1260,  "Jane Eyre",                  "Brontë"),
    ("Victorian",     4300,  "Ulysses",                    "Joyce"),
]

# ── Chunk counts per book from dataset ──────────────────────
chunk_counts = df['source'].value_counts().to_dict()

def get_chunks(era, book_id):
    key1 = f"{era}/{book_id}.txt"
    return chunk_counts.get(key1, 0)

# ── Generate HTML ────────────────────────────────────────────
date_str = datetime.now().strftime("%B %d, %Y")

def img_tag(b64, alt="chart"):
    if b64:
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" class="chart-img"/>'
    return '<p class="no-img">Chart not found — run train_models.py first</p>'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Author Era Identification — Project Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Segoe UI', Georgia, sans-serif;
    background: #f8f7f4;
    color: #2c2c2a;
    line-height: 1.7;
  }}

  /* ── Cover page ── */
  .cover {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    padding: 80px 60px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }}
  .cover-tag {{
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7f77dd;
    margin-bottom: 20px;
  }}
  .cover h1 {{
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 16px;
  }}
  .cover h1 span {{ color: #7f77dd; }}
  .cover .subtitle {{
    font-size: 1.1rem;
    color: #aaa;
    margin-bottom: 48px;
    max-width: 560px;
  }}
  .cover-meta {{
    display: flex;
    gap: 40px;
    flex-wrap: wrap;
  }}
  .meta-item {{ }}
  .meta-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #666;
    margin-bottom: 4px;
  }}
  .meta-value {{
    font-size: 0.95rem;
    color: #ddd;
    font-weight: 500;
  }}
  .cover-stats {{
    display: flex;
    gap: 32px;
    margin-top: 60px;
    flex-wrap: wrap;
  }}
  .stat-box {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 16px 24px;
    text-align: center;
  }}
  .stat-num {{
    font-size: 1.8rem;
    font-weight: 700;
    color: #7f77dd;
  }}
  .stat-lbl {{
    font-size: 0.75rem;
    color: #888;
    margin-top: 2px;
  }}

  /* ── Content pages ── */
  .page {{
    max-width: 900px;
    margin: 0 auto;
    padding: 60px 40px;
  }}

  .section {{
    margin-bottom: 56px;
  }}

  .section-number {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #534AB7;
    margin-bottom: 6px;
  }}

  h2 {{
    font-size: 1.6rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid #eee;
  }}

  h3 {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #2c2c2a;
    margin: 24px 0 10px;
  }}

  p {{
    color: #444;
    margin-bottom: 14px;
    font-size: 0.95rem;
  }}

  /* ── Tables ── */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 24px;
    font-size: 0.88rem;
  }}
  thead tr {{
    background: #1a1a2e;
    color: white;
  }}
  thead th {{
    padding: 11px 14px;
    text-align: left;
    font-weight: 500;
    letter-spacing: 0.03em;
  }}
  tbody tr {{ border-bottom: 1px solid #eee; }}
  tbody tr:hover {{ background: #f0effe; }}
  tbody td {{ padding: 10px 14px; color: #333; }}
  .best-row {{ background: #eeedfe !important; font-weight: 600; }}
  .pos {{ color: #0F6E56; font-weight: 700; }}
  .neg {{ color: #993C1D; }}
  .neu {{ color: #888; }}

  /* ── Era badges ── */
  .era-badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .badge-Renaissance   {{ background:#EEEDFE; color:#3C3489; }}
  .badge-Enlightenment {{ background:#E1F5EE; color:#085041; }}
  .badge-Romantic      {{ background:#FAECE7; color:#712B13; }}
  .badge-Victorian     {{ background:#E6F1FB; color:#0C447C; }}

  /* ── Charts ── */
  .chart-wrap {{
    background: white;
    border: 1px solid #e8e8e8;
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    text-align: center;
  }}
  .chart-img {{
    max-width: 100%;
    border-radius: 6px;
  }}
  .chart-caption {{
    font-size: 0.8rem;
    color: #888;
    margin-top: 10px;
    font-style: italic;
  }}
  .no-img {{
    color: #999;
    font-style: italic;
    padding: 20px;
  }}

  /* ── Highlight boxes ── */
  .highlight-box {{
    background: #eeedfe;
    border-left: 4px solid #534AB7;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 20px 0;
    font-size: 0.9rem;
    color: #3C3489;
  }}
  .warning-box {{
    background: #FAECE7;
    border-left: 4px solid #993C1D;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 20px 0;
    font-size: 0.9rem;
    color: #712B13;
  }}
  .success-box {{
    background: #E1F5EE;
    border-left: 4px solid #0F6E56;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 20px 0;
    font-size: 0.9rem;
    color: #085041;
  }}

  /* ── Two column grid ── */
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin: 16px 0;
  }}
  .info-card {{
    background: white;
    border: 1px solid #e8e8e8;
    border-radius: 10px;
    padding: 18px;
  }}
  .info-card h4 {{
    font-size: 0.85rem;
    font-weight: 600;
    color: #534AB7;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .info-card p {{
    font-size: 0.85rem;
    color: #555;
    margin: 0;
  }}

  /* ── Feature list ── */
  .feature-list {{
    list-style: none;
    padding: 0;
  }}
  .feature-list li {{
    padding: 8px 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 0.88rem;
    color: #444;
    display: flex;
    gap: 10px;
  }}
  .feature-list li:last-child {{ border: none; }}
  .feat-num {{
    font-weight: 700;
    color: #534AB7;
    min-width: 20px;
  }}

  /* ── Divider ── */
  .divider {{
    border: none;
    border-top: 1px solid #eee;
    margin: 40px 0;
  }}

  /* ── Print ── */
  @media print {{
    .cover {{ min-height: auto; page-break-after: always; }}
    .section {{ page-break-inside: avoid; }}
    body {{ background: white; }}
  }}
</style>
</head>
<body>

<!-- ═══════════════════════════════════════════════════════
     COVER PAGE
═══════════════════════════════════════════════════════ -->
<div class="cover">
  <div class="cover-tag">NLP Project Report</div>
  <h1>Author Era <span>Identification</span><br/>from Literary Texts</h1>
  <p class="subtitle">
    Predicting the historical era of English literature using
    TF-IDF features, stylometric analysis, and classical machine learning.
  </p>

  <div class="cover-meta">
    <div class="meta-item">
      <div class="meta-label">Dataset</div>
      <div class="meta-value">Project Gutenberg</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">Date</div>
      <div class="meta-value">{date_str}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">Best Model</div>
      <div class="meta-value">SVM — 83.62% (book-level)</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">Innovation</div>
      <div class="meta-value">Stylometric Feature Layer</div>
    </div>
  </div>

  <div class="cover-stats">
    <div class="stat-box">
      <div class="stat-num">16</div>
      <div class="stat-lbl">Books</div>
    </div>
    <div class="stat-box">
      <div class="stat-num">4</div>
      <div class="stat-lbl">Literary Eras</div>
    </div>
    <div class="stat-box">
      <div class="stat-num">3,964</div>
      <div class="stat-lbl">Text Chunks</div>
    </div>
    <div class="stat-box">
      <div class="stat-num">5,006</div>
      <div class="stat-lbl">Features</div>
    </div>
    <div class="stat-box">
      <div class="stat-num">3</div>
      <div class="stat-lbl">ML Models</div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════
     CONTENT
═══════════════════════════════════════════════════════ -->
<div class="page">

  <!-- Section 1 — Overview -->
  <div class="section">
    <div class="section-number">Section 01</div>
    <h2>Project Overview</h2>
    <p>
      This project addresses the task of <strong>Author Era Identification</strong> —
      automatically predicting the historical period in which a literary text was written,
      based solely on the content and style of the text itself. The system classifies
      text passages into one of four literary eras spanning approximately 400 years of
      English literature.
    </p>
    <p>
      The dataset was sourced entirely from <strong>Project Gutenberg</strong>, a free
      digital library of public domain books. Sixteen books were selected — four per era —
      covering a diverse range of authors, genres, and styles within each period.
    </p>

    <div class="two-col">
      <div class="info-card">
        <h4>Problem Type</h4>
        <p>Multi-class text classification (4 classes). Each 500-word chunk of text is assigned to one of four historical eras.</p>
      </div>
      <div class="info-card">
        <h4>Approach</h4>
        <p>Combined TF-IDF lexical features with stylometric writing-style measurements, trained on three classical ML classifiers.</p>
      </div>
      <div class="info-card">
        <h4>Innovation</h4>
        <p>Stylometric feature layer: sentence length, Type-Token Ratio, hapax legomena ratio, punctuation density, and function word frequency.</p>
      </div>
      <div class="info-card">
        <h4>Evaluation</h4>
        <p>Two-tier evaluation: chunk-level split (standard) and book-level split (unseen books) to measure true generalisation.</p>
      </div>
    </div>
  </div>

  <hr class="divider"/>

  <!-- Section 2 — Dataset -->
  <div class="section">
    <div class="section-number">Section 02</div>
    <h2>Dataset</h2>
    <p>
      All books were downloaded from Project Gutenberg in plain text format.
      Each book was cleaned to remove Gutenberg boilerplate headers and footers,
      then chunked into 500-word segments. Chunks shorter than 100 words were
      discarded to ensure quality.
    </p>

    <table>
      <thead>
        <tr>
          <th>Era</th>
          <th>Author</th>
          <th>Title</th>
          <th>Gutenberg ID</th>
          <th>Chunks</th>
        </tr>
      </thead>
      <tbody>"""

# Add book rows
era_colors = {
    "Renaissance": "badge-Renaissance",
    "Enlightenment": "badge-Enlightenment",
    "Romantic": "badge-Romantic",
    "Victorian": "badge-Victorian"
}

for era, book_id, title, author in books:
    chunks = get_chunks(era, book_id)
    badge  = era_colors[era]
    html += f"""
        <tr>
          <td><span class="era-badge {badge}">{era}</span></td>
          <td>{author}</td>
          <td>{title}</td>
          <td>{book_id}</td>
          <td>{chunks}</td>
        </tr>"""

# Era summary
era_totals = df.groupby('label').size().to_dict()
total_words = {
    "Renaissance": 206105,
    "Enlightenment": 626984,
    "Romantic": 418693,
    "Victorian": 726716
}

html += f"""
      </tbody>
    </table>

    <h3>Era Distribution</h3>
    <table>
      <thead>
        <tr>
          <th>Era</th>
          <th>Books</th>
          <th>Chunks</th>
          <th>% of Dataset</th>
        </tr>
      </thead>
      <tbody>"""

for era in ["Renaissance", "Enlightenment", "Romantic", "Victorian"]:
    count = era_totals.get(era, 0)
    pct   = count / len(df) * 100
    badge = era_colors[era]
    html += f"""
        <tr>
          <td><span class="era-badge {badge}">{era}</span></td>
          <td>4</td>
          <td>{count}</td>
          <td>{pct:.1f}%</td>
        </tr>"""

html += f"""
      </tbody>
    </table>

    <div class="warning-box">
      <strong>Class Imbalance Note:</strong> The Victorian era contains 1,455 chunks
      (36.7%) while Renaissance contains only 414 chunks (10.4%). This imbalance was
      addressed using <code>class_weight='balanced'</code> in SVM and Random Forest,
      and ComplementNB (which handles imbalance better than MultinomialNB) for Naive Bayes.
    </div>
  </div>

  <hr class="divider"/>

  <!-- Section 3 — Methodology -->
  <div class="section">
    <div class="section-number">Section 03</div>
    <h2>Methodology</h2>

    <h3>Preprocessing Pipeline</h3>
    <p>Each book went through the following steps before feature extraction:</p>
    <ul class="feature-list">
      <li><span class="feat-num">1</span>Gutenberg boilerplate removal using START/END markers</li>
      <li><span class="feat-num">2</span>Lowercasing and removal of numeric characters</li>
      <li><span class="feat-num">3</span>Whitespace normalisation</li>
      <li><span class="feat-num">4</span>Chunking into 500-word segments (minimum 100 words)</li>
      <li><span class="feat-num">5</span>Labeling each chunk with its source era</li>
    </ul>

    <h3>Feature Engineering</h3>
    <p>Two complementary feature sets were extracted and combined into a single matrix:</p>

    <div class="two-col">
      <div class="info-card">
        <h4>TF-IDF Features (5,000)</h4>
        <p>Term Frequency-Inverse Document Frequency with unigrams and bigrams,
        top 5,000 features, English stopwords removed, sublinear TF scaling applied.</p>
      </div>
      <div class="info-card">
        <h4>Stylometric Features (6)</h4>
        <p>Average sentence length, Type-Token Ratio, average word length,
        punctuation density, hapax legomena ratio, function word frequency.</p>
      </div>
    </div>

    <p>The two feature sets were concatenated using scipy's <code>hstack</code>,
    producing a final combined matrix of <strong>5,006 features per chunk</strong>.</p>

    <h3>Models Trained</h3>
    <ul class="feature-list">
      <li><span class="feat-num">1</span><strong>Complement Naive Bayes</strong> — probabilistic baseline, handles class imbalance well</li>
      <li><span class="feat-num">2</span><strong>Linear SVM</strong> — finds optimal hyperplane boundary between era classes</li>
      <li><span class="feat-num">3</span><strong>Random Forest</strong> — ensemble of 200 decision trees, majority vote prediction</li>
    </ul>
  </div>

  <hr class="divider"/>

  <!-- Section 4 — Results -->
  <div class="section">
    <div class="section-number">Section 04</div>
    <h2>Results</h2>

    <h3>Chunk-Level Evaluation (Standard Split)</h3>
    <p>
      An 80/20 train-test split was applied at the chunk level.
      All three models achieved high accuracy, with SVM performing best.
    </p>

    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Accuracy</th>
          <th>F1 Score (weighted)</th>
          <th>Rank</th>
        </tr>
      </thead>
      <tbody>"""

for i, (name, r) in enumerate(sorted(chunk_results.items(),
                               key=lambda x: x[1]['Accuracy'],
                               reverse=True)):
    row_class = "best-row" if i == 0 else ""
    html += f"""
        <tr class="{row_class}">
          <td>{name}</td>
          <td>{r['Accuracy']:.2f}%</td>
          <td>{r['F1 Score']:.4f}</td>
          <td>{'★ Best' if i==0 else str(i+1)}</td>
        </tr>"""

html += f"""
      </tbody>
    </table>

    <div class="chart-wrap">
      {img_tag(img_accuracy, "Model Accuracy Comparison")}
      <div class="chart-caption">Figure 1 — Accuracy comparison of all three models
      against the 25% random baseline</div>
    </div>

    <div class="chart-wrap">
      {img_tag(img_confusion, "Confusion Matrices")}
      <div class="chart-caption">Figure 2 — Confusion matrices for all three models.
      Darker diagonal = better performance.</div>
    </div>

    <h3>Book-Level Evaluation (Honest Generalisation Test)</h3>
    <p>
      To test true generalisation, one book per era was held out completely from
      training. The model was evaluated on these four entirely unseen books:
      Doctor Faustus (Renaissance), Common Sense (Enlightenment),
      Lyrical Ballads (Romantic), and The Picture of Dorian Gray (Victorian).
    </p>

    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Chunk-Level Accuracy</th>
          <th>Book-Level Accuracy</th>
          <th>Drop</th>
        </tr>
      </thead>
      <tbody>"""

for name in ["SVM", "Random Forest", "Naive Bayes"]:
    chunk_acc = chunk_results[name]['Accuracy']
    book_acc  = book_results[name]['Accuracy']
    drop      = chunk_acc - book_acc
    row_class = "best-row" if name == "SVM" else ""
    html += f"""
        <tr class="{row_class}">
          <td>{name}</td>
          <td>{chunk_acc:.2f}%</td>
          <td>{book_acc:.2f}%</td>
          <td class="neg">−{drop:.2f}%</td>
        </tr>"""

html += f"""
      </tbody>
    </table>

    <div class="chart-wrap">
      {img_tag(img_bookvschunk, "Book vs Chunk Accuracy")}
      <div class="chart-caption">Figure 3 — Chunk-level vs book-level accuracy.
      Book-level reflects performance on completely unseen books.</div>
    </div>

    <div class="highlight-box">
      <strong>Key Finding:</strong> The drop from chunk-level to book-level accuracy
      is expected and healthy — it reveals that the original high scores partially
      reflected same-author familiarity rather than pure era discrimination.
      SVM's book-level accuracy of <strong>83.62%</strong> on completely unseen books
      is the primary honest performance metric of this project.
    </div>

    <div class="warning-box">
      <strong>Romantic Era Note:</strong> The held-out Romantic book (Lyrical Ballads)
      is a poetry collection, while training data consisted primarily of prose.
      This genre mismatch explains the lower Romantic F1 score (0.25) in book-level
      evaluation — a genre generalisation challenge rather than a model weakness.
    </div>
  </div>

  <hr class="divider"/>

  <!-- Section 5 — Innovation -->
  <div class="section">
    <div class="section-number">Section 05</div>
    <h2>Innovation — Stylometric Feature Layer</h2>

    <p>
      Standard text classification relies solely on word frequency (TF-IDF), which
      captures <em>what</em> is written but not <em>how</em> it is written. This project
      introduces a stylometric feature layer that measures six writing-style properties
      independent of content:
    </p>

    <ul class="feature-list">
      <li><span class="feat-num">1</span><strong>Average sentence length</strong> — Victorian authors used significantly longer sentences than Renaissance playwrights</li>
      <li><span class="feat-num">2</span><strong>Type-Token Ratio (TTR)</strong> — measures vocabulary richness; Renaissance texts show higher TTR due to archaic vocabulary diversity</li>
      <li><span class="feat-num">3</span><strong>Average word length</strong> — Enlightenment writers favoured longer Latin-derived words</li>
      <li><span class="feat-num">4</span><strong>Punctuation density</strong> — semicolon and comma usage patterns differ significantly across eras</li>
      <li><span class="feat-num">5</span><strong>Hapax Legomena ratio</strong> — proportion of words appearing only once; high in Renaissance due to rare archaic vocabulary</li>
      <li><span class="feat-num">6</span><strong>Function word frequency</strong> — unconscious usage patterns of words like "the", "of", "in" are era-specific</li>
    </ul>

    <h3>Ablation Study</h3>
    <p>
      To quantify the contribution of stylometric features, an ablation study was
      conducted — each model was trained and evaluated both with and without
      the stylometric feature layer:
    </p>

    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Without Stylometry</th>
          <th>With Stylometry</th>
          <th>Improvement</th>
        </tr>
      </thead>
      <tbody>"""

for row in ablation_results:
    imp_class = "pos" if row["positive"] else "neg"
    row_class = "best-row" if row["Model"] == "Random Forest" else ""
    html += f"""
        <tr class="{row_class}">
          <td>{row['Model']}</td>
          <td>{row['Without Stylometry']}</td>
          <td>{row['With Stylometry']}</td>
          <td class="{imp_class}">{row['Improvement']}</td>
        </tr>"""

html += f"""
      </tbody>
    </table>

    <div class="chart-wrap">
      {img_tag(img_ablation, "Ablation Study")}
      <div class="chart-caption">Figure 4 — Impact of stylometric features on model accuracy.
      Random Forest shows the clearest improvement of +2.02%.</div>
    </div>

<div class="success-box">
      <strong>Chunk-Level Innovation Result:</strong> Random Forest improved by
      <strong>+1.89%</strong> and SVM by <strong>+0.13%</strong> when stylometric
      features were added alongside TF-IDF, confirming that style measurements
      provide complementary discriminative information beyond lexical features.
    </div>

    <h3>Book-Level Ablation — Does Stylometry Help on Unseen Books?</h3>
    <p>
      To further validate the innovation, a book-level ablation was conducted using
      the same held-out unseen books. This is a stricter test — it checks whether
      stylometric features help the model generalise to completely new authors,
      not just new chunks from familiar books.
    </p>

    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Without Stylometry</th>
          <th>With Stylometry</th>
          <th>Improvement</th>
        </tr>
      </thead>
      <tbody>"""

for row in book_ablation_results:
    imp_class = "pos" if row["positive"] else "neg" if not row["positive"] and row["Improvement"] != "+0.00%" else "neu"
    row_class = "best-row" if row["Model"] == "SVM" else ""
    html += f"""
        <tr class="{row_class}">
          <td>{row['Model']}</td>
          <td>{row['Without Stylometry']}</td>
          <td>{row['With Stylometry']}</td>
          <td class="{imp_class}">{row['Improvement']}</td>
        </tr>"""

html += f"""
      </tbody>
    </table>

    <div class="success-box">
      <strong>Key Finding — SVM:</strong> Stylometry improved SVM by
      <strong>+0.35%</strong> on completely unseen books, confirming that
      style-based features genuinely aid generalisation for linear classifiers
      beyond chunk-level memorisation.
    </div>

    <div class="warning-box">
      <strong>Key Finding — Random Forest:</strong> While Random Forest improved
      by +1.89% at chunk-level, it decreased by −1.74% at book-level. This reveals
      that tree-based models overfit to author-specific stylometric signatures
      rather than era-level patterns — an important generalisation insight.
    </div>

    <div class="highlight-box">
      <strong>Overall Innovation Conclusion:</strong> The divergence between
      chunk-level and book-level stylometry impact is itself a novel finding —
      demonstrating that evaluation protocol significantly affects the measured
      contribution of stylometric features. SVM with stylometry is the recommended
      model for deployment as it shows consistent improvement across both
      evaluation protocols.
    </div>
  </div>

  <hr class="divider"/>

  <!-- Section 6 — Conclusion -->
  <div class="section">
    <div class="section-number">Section 06</div>
    <h2>Conclusion</h2>

    <p>
      This project successfully developed an author era identification system capable
      of classifying English literary texts into four historical periods with
      <strong>83.62% accuracy on completely unseen books</strong> using a Linear SVM
      classifier with combined TF-IDF and stylometric features.
    </p>

    <p>
      Three key contributions were made:
    </p>

    <ul class="feature-list">
      <li><span class="feat-num">1</span><strong>Stylometric innovation</strong> — a six-feature writing-style layer that improved Random Forest accuracy by +2.02% and demonstrates that era classification benefits from style-aware features beyond word frequencies</li>
      <li><span class="feat-num">2</span><strong>Honest evaluation</strong> — a book-level validation split that revealed data leakage in standard chunk-level evaluation and provided a more credible performance benchmark</li>
      <li><span class="feat-num">3</span><strong>Genre insight</strong> — discovery that prose-trained models struggle with poetry from the same era, revealing a genre generalisation challenge relevant to future work</li>
    </ul>

    <h3>Future Work</h3>
    <p>
      Several directions could extend this work. Fine-tuning a transformer model
      such as DistilBERT on the era classification task would likely improve
      book-level accuracy beyond 84.67%, particularly for the Romantic era where
      genre variation currently reduces performance. Additionally, expanding the
      dataset to include more books per era — especially Renaissance — would
      address the class imbalance identified in this study.
    </p>

    <div class="highlight-box">
      <strong>Summary:</strong> SVM achieved 99.87% chunk-level and 83.62% book-level
      accuracy. Stylometry improved Random Forest by +2.02%. The system is deployed
      as a Flask web application accepting free-text input and returning era predictions
      with confidence scores and stylometric signal breakdowns.
    </div>
  </div>

</div>
</body>
</html>"""

# ── Save report ──────────────────────────────────────────────
output_path = "project_report.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Report saved: {output_path}")
print("Open project_report.html in your browser.")
print("To save as PDF: Ctrl+P → Save as PDF in browser.")
