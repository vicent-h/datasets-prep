import unidecode


class UnidecodeNorm:
    def __init__(self):
        pass

    def apply(self, text: str) -> str:
        """
        Normalize the input text using unidecode.

        Args:
            text (str): The input text to be normalized.

        Returns:
            str: The normalized text.
        """
        return unidecode.normalize("NFCK", text), True