import re
from typing import List
from nltk.stem import PorterStemmer


class Tokenizer:
    def __init__(self, stop_words: List[str] = None, use_stemming: bool = True):
        self.stop_words = set(stop_words) if stop_words else set()
        self.stemmer = PorterStemmer() if use_stemming else None

    def tokenize(self, text: str) -> List[str]:
        # 1. Lowercase
        text = text.lower()

        # 2. Handle hyphens (important for SciFact)
        text = text.replace("-", " ")

        # 3. Keep only alphanumeric
        tokens = re.findall(r'\b[a-z0-9]+\b', text)

        # 4. Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]

        # 5. Stemming
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens