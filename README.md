# 🔍 Search Engine from Scratch

A Python-based search engine project demonstrating foundational search algorithms and data structures from the ground up.

This project is built using lightweight, standard Python libraries to deeply understand how search engines work under the hood. It intentionally avoids high-level, out-of-the-box solutions (like LangChain or Elasticsearch) to expose the underlying math, data structures, and algorithms—from Inverted Indices and Okapi BM25 to Vector Embeddings and Reciprocal Rank Fusion.

## 🗺️ Project Structure

```
rag-search-engine/
├── Level1-keyword-search/ # Keyword search implementations
├── data/
│   ├── movies.json            # Movie dataset
│   └── stop_words.txt         # List of stop words to filter out
├── cache/                     # Directory for storing the built index
├── main.py                    # Simple script (currently minimal)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## 🚀 Level 1: [Keyword Search](./Level1-keyword-search) (Current Features)
- **Inverted Index**: Efficiently maps terms to documents for fast retrieval.
- **Keyword Search**: Search for movies based on keywords in their title or description.
- **Term Frequency (TF)**: Calculate how frequently a term appears in a specific document.
- **Inverse Document Frequency (IDF)**: Calculates the importance of a term across the entire corpus.
- **TF-IDF Score**: Compute the TF-IDF score to evaluate the relevance of a term to a specific document.
- **Text Preprocessing**: Tokenization, stop word removal, and stemming (using PorterStemmer) are applied to text data.
- **BM25 TF & IDF**: Implements components of the Okapi BM25 ranking function, including its variants of term frequency saturation and inverse document frequency.
- **BM25 Search**: Ranks search results using the Okapi BM25 algorithm for improved relevance.
### When Keyword Search Is Better?
 - Exact terms: Medical terms like "COVID-19"
 - Proper nouns: Specific titles like "The Matrix"
 - Technical jargon: Programming terms like "dependency injection"

## 🚀 Level 2: [Semantic Search](./Level2-semantic-search) (Current Features)

**Keyword search has serious limitations: it can only match when a keyword explicitly appears in the target document.**

Semantic search, on the other hand, understands the *meaning* behind the words. It finds relevant documents even when they don't contain your exact search terms by matching concepts rather than just spelling.

### Keyword vs. Semantic Search

**Query:** `"exciting adventure"`

* 🔴 **Keyword Search**
    * **Searches for:** Exact matches of "exciting" AND "adventure".
    * **Misses:** Films described as a "thrilling journey".
* 🟢 **Semantic Search**
    * **Understands:** You want action-packed films.
    * **Finds:** Movies described with words like "gripping", "suspenseful", or "journey".

### When Semantic Search Excels?

Semantic search is incredibly powerful for handling the messy, intent-driven way humans actually search:

* **Synonym Matching:** A search for *"happy movies"* easily finds films described as "joyful" or "uplifting".
* **Conceptual Queries:** A search for *"movies about friendship"* finds films exploring character bonds, even if the word "friendship" is never used.
* **Natural Language:** A search like *"What bear movies are good for kids?"* understands the intent for family-friendly animal content, rather than just matching the words "bear", "good", and "kids".
## 🧠 Why Build This?

Modern AI tooling often treats search as a black box (e.g., simply calling `.as_retriever()` in LangChain). By building these algorithms from scratch, this project demystifies the search process. You learn *why* common words drown out rare words in basic TF, *how* BM25 penalizes excessively long documents, and eventually, *where* lexical search fails and semantic search thrives.
