import re
from typing import List, Iterable

# pyrefly: ignore [missing-import]
from nltk.stem import PorterStemmer

# especially used for this medical terms dataset
TOKEN_RE = re.compile(r"""
    (?:[a-z]+(?:[-/][a-z0-9]+)+\+?) |   # hyphen/slash compounds, optional trailing +
    (?:[a-z]+[0-9]+\+?)              |   # alnum like il6, cd4+
    (?:[0-9]+(?:\.[0-9]+)?)          |   # integers/decimals
    (?:[a-z]+)                           # plain words
""", re.VERBOSE)


class Tokenizer:
    def __init__(self, stop_words: List[str] = None, use_stemming: bool = False,
                 emit_parts: bool = True):

        self.stop_words = set(stop_words) if stop_words else set()
        self.stemmer = PorterStemmer() if use_stemming else None
        self.emit_parts = emit_parts

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        #Lexical Analysis / Token Extraction (segmentation)
        tokens = TOKEN_RE.findall(text)
        # optionally add parts for hyphen/slash compounds
        if self.emit_parts:
            extra = []
            for t in tokens:
                if '-' in t or '/' in t:
                    extra.extend(re.split(r"[-/]", t))
            tokens = tokens + extra
        # remove empties and stopwords
        tokens = [t for t in tokens if t and (t not in self.stop_words)]
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

        