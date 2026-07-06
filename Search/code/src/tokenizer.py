import re
from typing import List, Iterable

# pyrefly: ignore [missing-import]
from nltk.stem import PorterStemmer

# especially used for this medical terms dataset
TOKEN_RE = re.compile(
    r"""
    (?:[a-z]+(?:[-/][a-z0-9]+)+\+?) |   # The Compound Matcher : hyphen/slash compounds, optional trailing +
    (?:[a-z]+[0-9]+\+?)             |   # The Alphanumeric Matcher: alnum like il6, cd4+
    (?:[0-9]+(?:\.[0-9]+)?)         |   # The Number Matcher: integers/decimals
    (?:[a-z]+)                           # The Word Matcher: plain words
""",
    re.VERBOSE,
)

class Tokenizer:
    def __init__(
        self,
        stop_words: List[str] = None,
        use_stemming: bool = False,
        emit_parts: bool = True,
    ):

        self.stop_words = set(stop_words) if stop_words else set()
        self.stemmer = PorterStemmer() if use_stemming else None
        # breaks hyphen/slash compounds into tokens (e.g., "covid-19" becomes "covid-19", "covid", and "19") if True.
        self.emit_parts = emit_parts

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # Lexical Analysis / Token Extraction (segmentation): Applies the regex to the string and extracts all matching patterns into a list of strings.
        tokens = TOKEN_RE.findall(text)
        # optionally add parts for hyphen/slash compounds : recall boost
        if self.emit_parts:
            extra = []
            for t in tokens:
                if "-" in t or "/" in t:
                    extra.extend(re.split(r"[-/]", t))
            tokens = tokens + extra
        # remove empties and stopwords
        tokens = [t for t in tokens if t and (t not in self.stop_words)]
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

# This is a Term-Based Tokenizer: