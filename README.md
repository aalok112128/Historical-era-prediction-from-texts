# Historical Era Prediction from Literary Texts

A complete NLP pipeline for predicting the **historical era of English literary texts** using **TF-IDF**, **stylometric features**, and **classical machine learning**.

This project classifies text into one of four literary periods:

- Renaissance
- Enlightenment
- Romantic
- Victorian

It is designed not just as a classifier, but as a small research-style system with:

- dataset construction from Project Gutenberg
- preprocessing and chunking
- lexical + stylometric feature engineering
- multiple model comparison
- honest book-level validation
- visual reports
- a Flask web app for live prediction

---

## Project Overview

Language changes over time. Vocabulary, sentence structure, punctuation habits, and writing style all evolve across centuries. This project uses those patterns to identify the likely era of a text passage.

The system combines:

- **TF-IDF features** to capture lexical patterns
- **Stylometric features** to capture writing style
- **Classical ML models** for efficient and interpretable classification

Unlike many simple text classification demos, this project also includes a stronger validation setup that checks whether the model generalizes to **completely unseen books**, not just unseen chunks from familiar books.

---

## Dataset

The corpus was built from **16 Project Gutenberg books**, with **4 books per era**.

### Eras Covered

- **Renaissance**
- **Enlightenment**
- **Romantic**
- **Victorian**

### Dataset Statistics

- **16 books**
- **4 literary eras**
- **3,964 text chunks**
- **500-word chunk size**
- **Minimum chunk length:** 100 words

### Chunk Distribution

- Victorian: 1455
- Enlightenment: 1256
- Romantic: 839
- Renaissance: 414

This class imbalance is handled during training using balanced model settings where appropriate.

---

## Pipeline

### 1. Data Collection
`download_books.py` downloads selected Project Gutenberg texts and organizes them by era.

### 2. Preprocessing
`preprocess.py` performs:

- Gutenberg header/footer removal
- text cleaning
- removal of noisy content
- chunking into 500-word segments
- labeling each chunk with its literary era and source book

The processed output is saved as `dataset.csv`.

### 3. Feature Extraction
`feature_extraction.py` builds two types of features:

#### TF-IDF Features
- 5000 features
- unigrams + bigrams
- English stopword removal
- sublinear term frequency scaling

#### Stylometric Features
Six handcrafted features are extracted:

- average sentence length
- type-token ratio
- average word length
- punctuation density
- hapax legomena ratio
- function word frequency

These are scaled and combined with TF-IDF to form a final **5006-dimensional feature space**.

### 4. Model Training
`train_models.py` trains and compares:

- Complement Naive Bayes
- Linear SVM
- Random Forest

It also generates:

- `results.json`
- `best_model.pkl`
- `best_model_name.txt`
- `accuracy_comparison.png`
- `confusion_matrices.png`

### 5. Validation
`validation_checks.py` performs stronger evaluation through:

- **book-level split**
- **stylometry ablation study**
- **chunk-level vs book-level comparison**

This is one of the strongest parts of the project because it checks whether the model truly learns era-level patterns rather than memorizing author-specific signals.

### 6. Reporting
`generate_reports.py` creates a styled HTML project report using the generated metrics and charts.

### 7. Web App
The Flask app in `webapp/` allows users to paste text and get:

- predicted literary era
- confidence across all eras
- stylometric signal breakdown

---

## Results

### Chunk-Level Evaluation
Standard 80/20 split on chunks:

- **SVM:** 99.87% accuracy
- **Naive Bayes:** 97.60% accuracy
- **Random Forest:** 96.97% accuracy

### Book-Level Evaluation
Testing on completely unseen books:

- **SVM:** 83.62% accuracy
- **Random Forest:** 78.05% accuracy
- **Naive Bayes:** 76.31% accuracy

### Key Insight
The drop from chunk-level to book-level accuracy is expected and important. It shows that chunk-level evaluation alone can overestimate performance when chunks from the same book appear in both training and testing.

The **book-level SVM result of 83.62%** is the most honest and meaningful performance number in this project.

---

## Ablation Study

The project also tests whether stylometric features actually help.

### Chunk-Level Improvement with Stylometry

- Naive Bayes: +0.00%
- SVM: +0.13%
- Random Forest: +1.39%

### Book-Level Improvement with Stylometry

- Naive Bayes: +0.00%
- SVM: +0.35%
- Random Forest: -1.74%

### Interpretation

Stylometric features help the **SVM** consistently, including on unseen books.  
For Random Forest, stylometry helps on chunk-level evaluation but hurts on book-level evaluation, suggesting overfitting to author-specific style rather than true era-level style.

This makes **SVM + TF-IDF + stylometry** the strongest overall model choice in the project.

---


## Project Structure

```text
NLP PROJECT/
├── download_books.py
├── preprocess.py
├── feature_extraction.py
├── train_models.py
├── validation_checks.py
├── generate_reports.py
├── literature_survey.md
├── requirements.txt
├── dataset.csv
├── results.json
├── validation_results.json
├── best_model.pkl
├── vectorizer.pkl
├── scaler.pkl
├── webapp/
│   ├── app.py
│   └── templates/
│       └── index.html
├── Renaissance/
├── Enlightenment/
├── Romantic/
└── Victorian/
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the books

```bash
python download_books.py
```

### 3. Preprocess and build dataset

```bash
python preprocess.py
```

### 4. Extract features

```bash
python feature_extraction.py
```

### 5. Train models

```bash
python train_models.py
```

### 6. Run validation checks

```bash
python validation_checks.py
```

### 7. Generate the report

```bash
python generate_reports.py
```

### 8. Launch the web app

```bash
python webapp/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## Web App Features

The web app allows users to paste a passage of at least 50 words and receive:

- predicted era
- relative confidence for all four eras
- era-specific metadata
- stylometric measurements from the input text

This makes the project easier to demonstrate in presentations and viva sessions.

---

## Limitations

Like any NLP project, this one has some limitations:

- the dataset is relatively small
- class distribution is uneven across eras
- genre differences can affect performance
- some era signals may overlap with author-specific signals
- poetry and prose may behave differently under the same era label

A clear example is the Romantic era, where genre mismatch can make generalization harder.

---

## Future Improvements

Possible next steps include:

- expanding the dataset with more books per era
- improving class balance
- separating genre-aware classification
- testing transformer-based models such as BERT or DistilBERT
- adding author-independent cross-validation
- improving interpretability with feature importance analysis
- deploying the web app publicly

---

## Final Takeaway

This project shows that **classical machine learning combined with stylometric analysis** can be highly effective for historical literary era prediction.

While chunk-level results are extremely high, the most valuable result is the **83.62% book-level accuracy with SVM**, which demonstrates strong generalization to unseen books.

It is a compact but meaningful NLP project that combines:

- literary computing
- stylometry
- text classification
- evaluation rigor
- practical deployment
