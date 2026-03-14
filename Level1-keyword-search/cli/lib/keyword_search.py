import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .constants import BM25_K1, BM25_B
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
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

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
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

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
        self.doc_lengths[doc_id] = len(tokens)


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
        """
        Calculates the TF-IDF score for a term in a given document.

        TF-IDF is a numerical statistic that is intended to reflect how important a
        word is to a document in a collection or corpus.

        Args:
            term: The term to calculate the TF-IDF score for.
            doc_id: The ID of the document.

        Returns:
            The TF-IDF score.
        """
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        idf = self.get_idf(term)
        tf = self.get_tf(doc_id, token)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        """
        Calculates the BM25 IDF for a term.
        equation : IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        N = total number of documents in the collection
        df = document frequency (how many documents contain this term)
        log = logarithm function (reduces the impact of very large numbers)
        :param term:
        :return: bm25Idf
        """
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        document_frequency = len(self.index[token])
        total_documents  =  len(self.docmap)
        idf = math.log(((total_documents - document_frequency + 0.5) / (document_frequency + 0.5)) +1 )
        return idf


    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B)->float:
        tf = self.get_tf(doc_id, term)
        # bm25_saturation = (tf * (k1 + 1)) / (tf + k1)
        # Length normalization factor
        # length_norm = 1 - b + b * (doc_length / avg_doc_length)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def __get_avg_doc_length(self) -> float:
        avg = 0.0
        if not self.doc_lengths:
            return avg
        for doc_length in self.doc_lengths.values():
            avg += doc_length
        return avg / len(self.doc_lengths)

    def bm25(self, doc_id, term):
        # BM25 = bm25_tf * bm25_idf
        bm25idf = self.get_bm25_idf(term)
        bm25tf = self.get_bm25_tf(doc_id, term)
        return bm25idf * bm25tf

    def bm25_search(self, query, limit):
        query_tokens = tokenize_text(query)
        score = {}
        for query_token in query_tokens:
            matching_doc_ids = self.get_documents(query_token)
            for doc_id in matching_doc_ids:
                if doc_id in score:
                    score[doc_id] += self.bm25(doc_id, query_token)
                    continue
                else:
                    score[doc_id] = self.bm25(doc_id, query_token)
        return sorted(score.items(), key=lambda x: x[1], reverse=True)[:limit]










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

    This is a simple implementation that returns documents that match any of the
    query tokens.

    Args:
        query: The search query string.
        limit: The maximum number of results to return.

    Returns:
        A list of matching document dictionaries, up to the specified limit.
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
    Command to get the TF-IDF score of a term in a document.

    Args:
        term: The term to query.
        doc_id: The document ID.

    Returns:
        The TF-IDF score.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_tfidf(term, doc_id)

def bm25_idf_command(term: str) -> float:
    """
    Command to get the BM25 IDF score of a term.

    Args:
        term: The term to query.

    Returns:
        The IDF score.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    """
    Command to get the bm25 term frequency of a term in a document.

    Args:
        doc_id: The document ID.
        term: The term to query.

    Returns:
        The term frequency.
        :param term:
        :param doc_id:
        :param b:
        :param k1:
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25search_command(query: str, limit=5):
    """
    Command to get the bm25 term frequency of a term in a document.

    Args:


    Returns:
        The term frequency.

    """
    idx = InvertedIndex()
    idx.load()
    bm25_result = idx.bm25_search(query, limit)
    i=1
    for result in bm25_result:
        docId = result[0]
        score = result[1]
        docTitle = idx.docmap[docId]['title']
        print(f"{i}. ({docId}) {docTitle} - Score: {score:.2f}")
