import unicodedata
from src.processor import Processor


class UnidecodeNorm(Processor):
    def __init__(self, format='NFC'):
        self.format = format

    def apply(self, text: str) -> str:
        """
        Normalize the input text using unidecode.

        Args:
            text (str): The input text to be normalized.

        Returns:
            str: The normalized text.
        """
        return unicodedata.normalize(self.format, text), True, {}

    def apply_pairs(self, text1, text2, **kwargs):

        """
        Normalize two input texts using unidecode.

        Args:
            text1 (str): The first input text to be normalized.
            text2 (str): The second input text to be normalized.

        Returns:
            tuple[str, str]: A tuple containing the normalized texts.
        """
        text1, _, _ = self.apply(text1)
        text2, _, _ = self.apply(text2)
        return (text1, text2), True, {}
