import json
import os
import string
from nltk.stem import PorterStemmer
import pickle
from collections import Counter
import math

BM25_K1 = 1.5
BM25_B = 0.75

stemmer = PorterStemmer()
table = str.maketrans("", "", string.punctuation)

stop_words_path = os.path.join(os.path.dirname(__file__), "../../data/stopwords.txt")
with open(stop_words_path, "r", encoding="utf-8") as f:
    stop_words = set(f.read().splitlines())


def tokenize(text):
    words = text.lower().translate(table).split()
    return [stemmer.stem(word) for word in words if word not in stop_words]

def tokenize_bigrams(text):
    words = tokenize(text)
    if len(words) < 2:
        return []
    return [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]


class InvertedIndex:
    def __init__(self, index_name="index") -> None:
        self.index = {}  # maps a word (token) to a set of movie ids it belongs to
        self.docmap = {}  # maps a movieId to it's title
        self.term_frequencies = {}  # maps document ids (movie object ids) to a counter for the number of tokens included
        self.doc_lengths = {}  # maps document ids to their lengths
        self.index_name = index_name

        self.movies_path = os.path.join(
            os.path.dirname(__file__), "../../data/movies.json"
        )
        self.index_path = os.path.join(
            os.path.dirname(__file__), f"../../cache/{index_name}.pkl"
        )

    def tokenize_strategy(self, text):
        return tokenize(text)

    def __add_document(self, doc_id, text):
        for word in self.tokenize_strategy(text):
            self.index.setdefault(word, set()).add(doc_id)
            self.term_frequencies.setdefault(doc_id, Counter())[word] += 1
            self.doc_lengths[doc_id] = self.doc_lengths.get(doc_id, 0) + 1

    def __get_avg_doc_length(self) -> float:
        return (
            (sum(self.doc_lengths.values()) / len(self.doc_lengths))
            if self.doc_lengths
            else 0.0
        )

    def get_documents(self, term):
        return sorted(list(self.index.get(term, set())))

    def build(self):
        with open(self.movies_path, "r", encoding="utf-8") as f:
            movies_list = json.load(f)["movies"]
        for movie in movies_list:
            self.__add_document(
                movie["id"], movie["title"] + " " + movie["description"]
            )
            self.docmap[movie["id"]] = movie

    def get_tf(self, doc_id, term):
        tokenized_term = self.tokenize_strategy(term)
        if len(tokenized_term) == 0:
            return 0
        token = tokenized_term[0]
        return self.term_frequencies.get(doc_id, {}).get(token, 0)

    def get_bm25_idf(self, term: str) -> float:
        """
        Calculate the BM25 inverse document frequency for a given term.
        Inverse document frequency (IDF) measures how important a term is across all documents. How unique or rare a term is.
        Higher IDF means the term is more unique, lower IDF means it is more common.

        How this works:
        - uses BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1) where
            - N is the total number of documents
            - df is the number of documents containing the term
        - Simply put it is the ratio of documents not containing the term to those containing it,
            adjusted with 0.5 to avoid division by zero.

        Args:
        term (str): The term to calculate the BM25 IDF for.

        Returns:
        float: The BM25 IDF value.

        """
        tokenized_term = self.tokenize_strategy(term)
        if len(tokenized_term) == 0:
            return 0.0
        token = tokenized_term[0]
        df = len(self.get_documents(token))
        N = len(self.docmap)
        if N == 0: return 0.0
        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        """
        Calculate the BM25 term frequency for a given term in a document.
        This is the frequency of the term in the document adjusted for document length and term saturation.
        How this works:
            - term frequency (tf) is scaled by document length and parameters k1 and b
            - length normalization is applied based on average document length
        Args:
            doc_id (int): The document id to calculate BM25 TF for.
            term (str): The term to calculate BM25 TF for
            k1 (float): The BM25 k1 parameter
            b (float): The BM25 b parameter
        Returns:
            float: The BM25 TF value
        """
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = (
            1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1
        )
        raw_tf = self.get_tf(doc_id, term)
        tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return tf

    def bm25(self, doc_id, term):
        """
        Calculate the BM25 score for a given term in a document.

        Args:
            doc_id (int): The document id to calculate BM25 score for.
            term (str): The term to calculate BM25 score for

        Returns:
            float: The BM25 score value. This is the product of BM25 TF and BM25 IDF.
            Simply put it represents the relevance of the term in the document adjusted for term frequency and document frequency.

        """
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query, limit) -> list:
        """
        Perform BM25 search for a given query and return top documents.
        Args:
            query (str): The search query.
            limit (int): The maximum number of results to return.
        Returns:
            list: List of tuples containing document and its BM25 score.
        """
        tokenized_query = self.tokenize_strategy(query)
        scores = {}
        for term in tokenized_query:
            docs = self.get_documents(term)
            if not docs:
                continue
                
            idf = self.get_bm25_idf(term)
            for doc_id in docs:
                tf = self.get_bm25_tf(doc_id, term)
                scores[doc_id] = scores.get(doc_id, 0) + (tf * idf)
                
        ranked_docs = list(
            map(
                lambda x: (self.docmap[x[0]], x[1]),
                sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit],
            )
        )
        return ranked_docs

    def save(self):
        os.makedirs("cache", exist_ok=True)
        base = os.path.dirname(self.index_path)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(base, f"{self.index_name}_docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        with open(os.path.join(base, f"{self.index_name}_term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(os.path.join(base, f"{self.index_name}_doc_lengths.pkl"), "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        base = os.path.dirname(self.index_path)
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(os.path.join(base, f"{self.index_name}_docmap.pkl"), "rb") as f:
            self.docmap = pickle.load(f)
        with open(os.path.join(base, f"{self.index_name}_term_frequencies.pkl"), "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(os.path.join(base, f"{self.index_name}_doc_lengths.pkl"), "rb") as f:
            self.doc_lengths = pickle.load(f)


class BigramInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        super().__init__(index_name="index_bigrams")
        
    def tokenize_strategy(self, text):
        # When looking up a single term (already tokenized as a bigram)
        if "_" in text and " " not in text:
            return [text]
        return tokenize_bigrams(text)


def bm25_idf_command(term: str) -> float:
    """
    Calculate the BM25 inverse document frequency for a given term.

    Args:
        term (str): The term to calculate the BM25 IDF for.
    Returns:
        float: The BM25 IDF value.
    """
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b1=BM25_B):
    """
    Calculate the BM25 term frequency for a given term in a document.

    Args:
        doc_id (int): The document id to calculate BM25 TF for.
        term (str): The term to calculate BM25 TF for
        k1 (float): The BM25 k1 parameter
        b1 (float): The BM25 b parameter
    Returns:
        float: The BM25 TF value
    """
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_tf(doc_id, term, k1, b1)


def bm25search_command(query: str, limit: int = 10) -> list:
    """
    Perform BM25 search for a given query and return top documents.
    Args:
        query (str): The search query.
        limit (int): The maximum number of results to return.
    Returns:
        list: List of tuples containing document id and its BM25 score.
    """
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.bm25_search(query, limit)

def bm25_search_bigrams(query: str, limit: int = 10) -> list:
    idx = BigramInvertedIndex()
    if not os.path.exists(idx.index_path):
        idx.build()
        idx.save()
    else:
        idx.load()
    return idx.bm25_search(query, limit)