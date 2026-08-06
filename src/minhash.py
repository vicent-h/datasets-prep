import re
import datasketch as ds


class MinHashDetector:
    """Detect near-duplicate sentences using MinHash + MinHashLSH.

    Methods
    - `is_duplicate(text)`: returns True if `text` is similar (>= threshold)
       to any previously seen sentence. If not duplicate, the sentence is
       added to the LSH index for future comparisons.
    """

    def __init__(self, num_perm=128, threshold=0.7):
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = ds.MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.hashes = {}  # key -> MinHash
        self.texts = {}   # key -> original normalized text
        self._counter = 0

    def _normalize(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text)

    def _minhash_from_text(self, text: str) -> ds.MinHash:
        mh = ds.MinHash(num_perm=self.num_perm)
        for token in self._normalize(text).split():
            mh.update(token.encode("utf8"))
        return mh

    def apply(self, text: str) -> bool:
        """Return True if `text` is near-duplicate of any stored sentence.

        If no near-duplicate is found the sentence is added to the index and
        the method returns False.
        """
        norm = self._normalize(text)
        mh = self._minhash_from_text(norm)
        candidates = self.lsh.query(mh)
        for key in candidates:
            other_mh = self.hashes.get(key)
            if other_mh is None:
                continue
            sim = mh.jaccard(other_mh)
            if sim >= self.threshold:
                return True
        # not duplicate -> add to index
        key = f"item{self._counter}"
        self._counter += 1
        self.hashes[key] = mh
        self.texts[key] = norm
        self.lsh.insert(key, mh)
        return text, False

    