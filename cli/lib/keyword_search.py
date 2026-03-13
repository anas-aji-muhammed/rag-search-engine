import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    """
    A class representing an inverted index for keyword search.

    The index maps tokens to the set of document IDs that contain them.
    It also stores document metadata and term frequencies.
    """
    def __init__(self) -> None:
        """
        Initializes the InvertedIndex with empty structures and file paths.
        """
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)

    def build(self) -> None:
        """
        Builds the inverted index from the movie data.

        Loads movies, processes each one, and adds it to the index.
        """
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        """
        Saves the index, document map, and term frequencies to disk.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        """
        Loads the index, document map, and term frequencies from disk.
        """
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        """
        Retrieves a list of document IDs that contain the given term.

        Args:
            term: The token to search for.

        Returns:
            A sorted list of document IDs.
        """
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Adds a document to the index.

        Args:
            doc_id: The unique identifier for the document.
            text: The text content of the document.
        """
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Gets the term frequency of a term in a specific document.

        Args:
            doc_id: The document ID.
            term: The term to look up.

        Returns:
            The number of times the term appears in the document.

        Raises:
            ValueError: If the term results in multiple tokens (or none).
        """
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        """
        Calculates the Inverse Document Frequency (IDF) for a term.

        Args:
            term: The term to calculate IDF for.

        Returns:
            The IDF score.

        Raises:
            ValueError: If the term results in multiple tokens (or none).
        """
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tfidf(self, term: str, doc_id: int) -> float:
#       calculate tf idf with TF-IDF = TF * IDF
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        idf = self.get_idf(term)
        tf = self.get_tf(doc_id, token)
        return tf * idf




def build_command() -> None:
    """
    Command to build and save the inverted index.
    """
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """
    Searches for documents matching the query using the inverted index.

    Args:
        query: The search query string.
        limit: The maximum number of results to return.

    Returns:
        A list of matching document dictionaries.
    """
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def preprocess_text(text: str) -> str:
    """
    Preprocesses text by converting to lowercase and removing punctuation.

    Args:
        text: The input text.

    Returns:
        The cleaned text.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes text by splitting, removing stopwords, and stemming.

    Args:
        text: The input text.

    Returns:
        A list of processed tokens.
    """
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


def tf_command(doc_id: int, term: str) -> int:
    """
    Command to get the term frequency of a term in a document.

    Args:
        doc_id: The document ID.
        term: The term to query.

    Returns:
        The term frequency.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    """
    Command to get the IDF score of a term.

    Args:
        term: The term to query.

    Returns:
        The IDF score.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tfidf_command(term: str, doc_id: int) -> float:
    """
    Command to get the IDF score of a term.

    Args:
        term: The term to query.

    Returns:
        The IDF score.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_tfidf(term, doc_id)
