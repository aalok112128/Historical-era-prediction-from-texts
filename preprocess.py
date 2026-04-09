# ============================================================
# Author Era Identification — Preprocessing Pipeline
# ============================================================

import os
import re
import nltk
import pandas as pd
from collections import Counter

# Download required NLTK data (only runs once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# ============================================================
# STEP 1 — DEFINE BOOK MAPPING
# ============================================================
# Maps each file ID to its era label
# This is how the code knows which era each book belongs to

book_map = {
    # Renaissance
    "Renaissance/1515.txt"  : "Renaissance",
    "Renaissance/15272.txt" : "Renaissance",
    "Renaissance/45988.txt" : "Renaissance",
    "Renaissance/779.txt"   : "Renaissance",

    # Enlightenment
    "Enlightenment/829.txt"  : "Enlightenment",
    "Enlightenment/521.txt"  : "Enlightenment",
    "Enlightenment/3300.txt" : "Enlightenment",
    "Enlightenment/147.txt"  : "Enlightenment",

    # Romantic
    "Romantic/1342.txt" : "Romantic",
    "Romantic/84.txt"   : "Romantic",
    "Romantic/9622.txt" : "Romantic",
    "Romantic/82.txt"   : "Romantic",

    # Victorian
    "Victorian/1400.txt" : "Victorian",
    "Victorian/174.txt"  : "Victorian",
    "Victorian/1260.txt" : "Victorian",
    "Victorian/4300.txt" : "Victorian",
}

# ============================================================
# STEP 2 — REMOVE GUTENBERG BOILERPLATE
# ============================================================
# Every Gutenberg file has a header and footer with legal text
# We strip everything outside the actual book content

def remove_boilerplate(text):
    """Remove Gutenberg header and footer from raw text."""

    # Gutenberg marks book start and end with these patterns
    start_patterns = [
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .+? \*\*\*",
        r"\*\*\*START OF THE PROJECT GUTENBERG EBOOK .+?\*\*\*",
    ]
    end_patterns = [
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .+? \*\*\*",
        r"\*\*\*END OF THE PROJECT GUTENBERG EBOOK .+?\*\*\*",
    ]

    # Find where actual book content starts
    start_pos = 0
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            break

    # Find where actual book content ends
    end_pos = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            end_pos = match.start()
            break

    return text[start_pos:end_pos].strip()


# ============================================================
# STEP 3 — CLEAN THE TEXT
# ============================================================
# Remove noise while keeping enough structure for stylometry

def clean_text(text):
    """Clean raw text — remove numbers, extra whitespace, odd characters."""

    # Remove numbers (they carry no stylistic meaning)
    text = re.sub(r'\d+', '', text)

    # Remove lines that are all caps (often chapter headings or metadata)
    lines = text.split('\n')
    lines = [l for l in lines if not (l.strip().isupper() and len(l.strip()) > 3)]
    text = '\n'.join(lines)

    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse multiple spaces into one
    text = re.sub(r' {2,}', ' ', text)

    # Remove weird characters but keep basic punctuation
    # We keep: letters, spaces, basic punctuation (.,;:!?'"-), newlines
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\'\"\-\n]', ' ', text)

    return text.strip()


# ============================================================
# STEP 4 — CHUNK THE TEXT
# ============================================================
# We split each book into 500-word chunks
# Each chunk = one training sample with the era label

def chunk_text(text, chunk_size=500, min_size=100):
    """Split text into chunks of roughly chunk_size words."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        # Only keep chunks with enough words
        if len(chunk.split()) >= min_size:
            chunks.append(chunk)

    return chunks


# ============================================================
# STEP 5 — LOAD, CLEAN, AND CHUNK ALL BOOKS
# ============================================================

def process_all_books(base_path="D:\\NLP PROJECT"):
    """Load all 16 books, clean them, chunk them, return a DataFrame."""

    all_chunks = []
    all_labels = []
    all_sources = []   # track which book each chunk came from

    print("Processing books...\n")

    for relative_path, era in book_map.items():
        full_path = os.path.join(base_path, relative_path.replace("/", "\\"))

        if not os.path.exists(full_path):
            print(f"  WARNING: File not found — {full_path}")
            continue

        # Load raw text
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()

        # Remove Gutenberg boilerplate
        clean = remove_boilerplate(raw_text)

        # Clean the text
        clean = clean_text(clean)

        # Chunk into 500-word pieces
        chunks = chunk_text(clean, chunk_size=500)

        all_chunks.extend(chunks)
        all_labels.extend([era] * len(chunks))
        all_sources.extend([relative_path] * len(chunks))

        book_id = relative_path.split("/")[1].replace(".txt", "")
        print(f"  {era:15} | ID {book_id:6} | {len(chunks):4} chunks | "
              f"{len(clean.split()):,} words after cleaning")

    # Build a DataFrame — one row per chunk
    df = pd.DataFrame({
        'text'  : all_chunks,
        'label' : all_labels,
        'source': all_sources
    })

    return df


# ============================================================
# STEP 6 — RUN AND SAVE
# ============================================================

df = process_all_books()

print("\n" + "="*55)
print("PREPROCESSING SUMMARY")
print("="*55)
print(f"Total chunks created : {len(df)}")
print(f"\nChunks per era:")
print(df['label'].value_counts().to_string())
print(f"\nSample chunk (first 200 chars):")
print(df['text'].iloc[0][:200])

# Save to CSV — this becomes your dataset for training
output_path = "D:\\NLP PROJECT\\dataset.csv"
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\nDataset saved to: {output_path}")
print("You are ready for feature extraction!")