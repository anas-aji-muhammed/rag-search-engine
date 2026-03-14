# Keyword Search Engine

A Python-based search engine project that implements keyword search functionality from scratch using an inverted index. This project is built using lightweight, standard Python libraries to demonstrate foundational search concepts. It allows users to build an index from movie data, search for movies, and calculate Term Frequency (TF), Inverse Document Frequency (IDF), and TF-IDF scores for specific terms.
```
Level1-keyword-search/
├── cli/
│   ├── keyword_search_cli.py  # Main CLI entry point for Level 1
│   └── lib/
│       ├── keyword_search.py  # Core search logic (InvertedIndex, TF, IDF, BM25)
│       ├── search_utils.py    # Utility functions (loading data, paths)
│       └── constants.py       # Project constants
└── README.md                # Documentation for Level 1
                # Project documentation
```

## Features

- **Inverted Index**: Efficiently maps terms to documents for fast retrieval.
- **Keyword Search**: Search for movies based on keywords in their title or description.
- **Term Frequency (TF)**: Calculate how frequently a term appears in a specific document.
- **Inverse Document Frequency (IDF)**: Calculates the importance of a term across the entire corpus.
- **TF-IDF Score**: Compute the TF-IDF score to evaluate the relevance of a term to a specific document.
- **Text Preprocessing**: Tokenization, stop word removal, and stemming (using PorterStemmer) are applied to text data.
- **BM25 TF & IDF**: Implements components of the Okapi BM25 ranking function, including its variants of term frequency saturation and inverse document frequency.
- **BM25 Search**: Ranks search results using the Okapi BM25 algorithm for improved relevance.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Download the movie dataset:
    Download the file from `https://njhkxqi5evlbap1x.public.blob.vercel-storage.com/movies.json` and save it to `data/movies.json`.

## Usage

The project provides a Command Line Interface (CLI) to interact with the search engine.

### 1. Build the Index

Before searching or calculating statistics, you must build the inverted index. This processes the `data/movies.json` file and saves the index to the `cache/` directory.

```bash
python cli/keyword_search_cli.py build
```

### 2. Search for Movies

Search for movies using a query string. The results will display the movie ID and title.

```bash
python cli/keyword_search_cli.py search "adventure movie"
```

### 3. Get Term Frequency (TF)

Find out how many times a specific term appears in a document (identified by its ID).

```bash
python cli/keyword_search_cli.py tf <doc_id> <term>
```

Example:
```bash
python cli/keyword_search_cli.py tf 1 "action"
```

### 4. Get Inverse Document Frequency (IDF)

Calculate the IDF score for a specific term, indicating its rarity across the dataset.

```bash
python cli/keyword_search_cli.py idf <term>
```

Example:
```bash
python cli/keyword_search_cli.py idf "rare_term"
```

### 5. Get TF-IDF Score

Calculate the TF-IDF score for a term in a specific document.

```bash
python cli/keyword_search_cli.py tfidf <term> <doc_id>
```

Example:
```bash
python cli/keyword_search_cli.py tfidf "action" 1
```

## Key Components

### `cli/lib/keyword_search.py`

Contains the `InvertedIndex` class which handles:
-   `build()`: Constructs the index from movie data.
-   `save()` / `load()`: Persists the index to disk using `pickle`.
-   `get_documents()`: Retrieves documents matching a term.
-   `get_tf()` / `get_idf()` / `get_tfidf()`: Computes TF, IDF, and TF-IDF scores. The `get_idf` function uses the BM25 IDF formula: `log((N - df + 0.5) / (df + 0.5) + 1)`.

Also includes text processing functions:
-   `tokenize_text()`: Splits text, removes stop words, and applies stemming.
-   `preprocess_text()`: Cleans text (lowercase, removes punctuation).

### `cli/lib/search_utils.py`

Helper functions for file handling and constant definitions (`CACHE_DIR`, `DATA_PATH`, etc.).

### `cli/keyword_search_cli.py`

The command-line interface wrapper that parses arguments and calls the appropriate functions from `keyword_search.py`.

## Implementing the Okapi BM25 Search Algorithm

This document outlines the theoretical concepts and practical implementation details for the Okapi BM25 search algorithm. BM25 is a state-of-the-art ranking function that improves upon the foundational TF-IDF (Term Frequency-Inverse Document Frequency) algorithm.

### Why BM25 is Better than TF-IDF

Standard TF-IDF suffers from three main issues that BM25 explicitly solves:

*   **Unstable IDF**: Basic IDF struggles with extreme values (division by zero, negative scores for common words, and overly high scores for rare words).
*   **Uncapped Term Frequency**: Basic TF scales linearly, allowing keyword stuffing (e.g., a document repeating "bear" 100 times dominating the results).
*   **No Length Normalization**: Longer documents naturally contain more words and get an unfair advantage over concise, highly relevant shorter documents.

### 1. Better IDF Calculation (Stable IDF)

The standard IDF formula is `log(N / df)`. BM25 modifies this to handle edge cases gracefully.

**BM25 IDF Formula:**
```
IDF = log((N - df + 0.5) / (df + 0.5) + 1)
```
*   **N** = Total number of documents in the collection.
*   **df** = Document frequency (number of documents containing the term).
*   **Numerator (N - df + 0.5)**: Count of documents without the term.
*   **Denominator (df + 0.5)**: Count of documents with the term.

**Why the constants?**
*   **+ 0.5 (Laplace Smoothing)**: Prevents division-by-zero errors when `df = 0`.
*   **+ 1**: Ensures the resulting IDF score is always positive, even for incredibly common terms.

**Implementation Note:**
Implemented via `get_bm25_idf(self, term: str) -> float`. Requires tokenizing the term and validating it as a single token before calculation.

### 2. Term Frequency (TF) Saturation

BM25 introduces diminishing returns for term frequency. After a certain point, finding the word again in the same document barely increases the score.

**Saturation Formula:**
```
tf_component = (tf * (k1 + 1)) / (tf + k1)
```
*   **tf**: Raw term frequency in the document.
*   **k1**: Tunable parameter controlling the saturation curve.

**The k1 Parameter:**
*   Typically defaults to `1.5` (`BM25_K1 = 1.5`).
*   A higher `k1` delays saturation (meaning the score keeps growing longer).
*   Because of the denominator (`tf + k1`), as `tf` approaches infinity, the whole component simply approaches `(k1 + 1)`. It mathematically cannot exceed this cap.

### 3. Document Length Normalization

To ensure long documents don't outrank short, highly-focused documents simply by having a higher word count, BM25 normalizes the TF component based on the document's length relative to the corpus average.

**Length Normalization Formula:**
```
length_norm = 1 - b + b * (doc_length / avg_doc_length)
```

**Updated TF Formula (with Normalization applied):**
```
tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
```
*   **doc_length**: The total token count of the specific document.
*   **avg_doc_length**: The average token count across all documents in the index.
*   **b**: Tunable parameter controlling the strength of length normalization.

**The b Parameter:**
*   Typically defaults to `0.75` (`BM25_B = 0.75`).
*   If `b = 1`, full normalization is applied.
*   If `b = 0`, normalization is turned off completely.

**Effect**: Long documents get a `length_norm > 1` (penalizing their score), while short documents get a `length_norm < 1` (boosting their score).

**Implementation Note:**
Requires tracking document lengths during indexing (`self.doc_lengths`), caching them, and calculating `avg_doc_length`.

### 4. The Final BM25 Search Execution

With the improved TF and IDF components built, the final scoring logic combines them.

**Single Term Score for a Document:**
```
BM25(doc, term) = get_bm25_tf(doc, term) * get_bm25_idf(term)
```

**Full Search Algorithm (`bm25_search`):**
1.  Tokenize the incoming search query.
2.  Initialize a tracking dictionary for document scores.
3.  Iterate through every document in the index:
    *   For each token in the query, calculate the `BM25(doc, term)` score.
    *   Sum the scores of all query tokens for that document.
4.  Sort the documents by their final accumulated score in descending order.
5.  Return the top N results (e.g., limited by a `--limit` CLI argument).

### Expected Output Format

When querying via CLI (e.g., `uv run cli/keyword_search_cli.py bm25search "animated family"`), results should map cleanly to Document IDs and scores:
```
1. (11342) Gakuen Alice - Score: 7.35
2. (30) Day of the Animals - Score: 7.14
3. (1043) Fantastic Mr. Fox - Score: 6.91
```
[bm25_equation]: ./bm25.png
![bm25_equation][bm25_equation]
