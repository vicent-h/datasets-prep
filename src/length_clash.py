import re
from src.processor import Processor

class LengthClash(Processor):
    def __init__(self, max_length_diff_ratio = 1.8):
        self.max_length_diff_ratio = max_length_diff_ratio
        self.lengths = {}

    def count_length(self, text):
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return len(text.split())

    def score(self, text1: str, text2: str) -> float:
        len1 = self.count_length(text1)
        len2 = self.count_length(text2)
        if len1 == 0 or len2 == 0:
            return 0.0
        length_diff_ratio = max(len1, len2) / min(len1, len2)
        return length_diff_ratio

    def apply_pairs(self, text1: str, text2: str, **kwargs) -> tuple:
        length_diff_ratio = self.score(text1, text2)
        eval = length_diff_ratio <= self.max_length_diff_ratio
        return (text1, text2), eval, {'length_diff_ratio': length_diff_ratio}