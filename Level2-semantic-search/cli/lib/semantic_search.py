import os
from sentence_transformers import SentenceTransformer
import numpy as np

from .common_utils import (
    CACHE_DIR,
    load_movies,
)

def add_vectors(v1: list[float], v2: list[float])->list[float]:
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    return [v1[i] + v2[i] for i in range(len(v1))]

def subtract_vectors(v1: list[float], v2: list[float])->list[float]:
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    return [v2[i] - v1[i] for i in range(len(v1))]

class SemanticSearch:
    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        return self.model.encode(text)

    def build_embeddings(self, documents):
        movie_descriptions = []
        for doc in documents:
            doc_description = f"{doc['title']} {doc['description']}"
            movie_descriptions.append(doc_description)
        self.embeddings = self.model.encode(movie_descriptions, show_progress_bar=True)
        np.save(f"{CACHE_DIR}/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(f"{CACHE_DIR}/movie_embeddings.npy"):
            self.embeddings = np.load(f"{CACHE_DIR}/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5 ):
        # Ensure embeddings have been loaded before searching.
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        # Generate the embedding for the user's query.
        query_embedding = self.generate_embedding(query)

        # Calculate the cosine similarity between the query embedding and each document embedding.
        scores = []
        for i, embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, embedding)
            # Store the score and the corresponding document as a tuple.
            scores.append((score, self.documents[i]))

        # Sort the scores in descending order to get the most similar documents first.
        scores.sort(key=lambda x: x[0], reverse=True)

        # Return the top 'limit' results.
        results = []
        for score, doc in scores[:limit]:
            results.append({
                "score": score,
                "title" : doc["title"],
                "description" : doc["description"]
            })
        return results



def verify_model():
    sm_search = SemanticSearch()
    print(f"Max sequence length: {sm_search.model.max_seq_length}")
    movies = load_movies()
    embeddings = sm_search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text: str):
    sm_search = SemanticSearch()
    embedding =  sm_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int = 5):
    sm_search = SemanticSearch()
    movies = load_movies()
    sm_search.load_or_create_embeddings(movies)
    results = sm_search.search(query, limit)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'].splitlines()[0]}...\n")