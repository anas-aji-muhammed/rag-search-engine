# RAG Search Engine

A Python-based search engine project that implements keyword search functionality using an inverted index. This project allows users to build an index from movie data, search for movies, and calculate Term Frequency (TF) and Inverse Document Frequency (IDF) for specific terms.

## Project Structure

```
rag-search-engine/
├── cli/
│   ├── keyword_search_cli.py  # Main CLI entry point
│   └── lib/
│       ├── keyword_search.py  # Core search logic (InvertedIndex, TF, IDF)
│       └── search_utils.py    # Utility functions (loading data, paths)
├── data/
│   ├── movies.json            # Movie dataset
│   └── stop_words.txt         # List of stop words to filter out
├── cache/                     # Directory for storing the built index
├── main.py                    # Simple script (currently minimal)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Features

- **Inverted Index**: Efficiently maps terms to documents for fast retrieval.
- **Keyword Search**: Search for movies based on keywords in their title or description.
- **Term Frequency (TF)**: Calculate how frequently a term appears in a specific document.
- **Inverse Document Frequency (IDF)**: Calculate the importance of a term across the entire corpus.
- **Text Preprocessing**: Tokenization, stop word removal, and stemming (using PorterStemmer) are applied to text data.

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

## Key Components

### `cli/lib/keyword_search.py`

Contains the `InvertedIndex` class which handles:
-   `build()`: Constructs the index from movie data.
-   `save()` / `load()`: Persists the index to disk using `pickle`.
-   `get_documents()`: Retrieves documents matching a term.
-   `get_tf()` / `get_idf()`: Computes TF and IDF scores.

Also includes text processing functions:
-   `tokenize_text()`: Splits text, removes stop words, and applies stemming.
-   `preprocess_text()`: Cleans text (lowercase, removes punctuation).

### `cli/lib/search_utils.py`

Helper functions for file handling and constant definitions (`CACHE_DIR`, `DATA_PATH`, etc.).

### `cli/keyword_search_cli.py`

The command-line interface wrapper that parses arguments and calls the appropriate functions from `keyword_search.py`.
