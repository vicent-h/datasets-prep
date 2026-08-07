class Processor:
    def __init__(self):
        pass
    def apply(self, text: str) -> str:
        """
        Process the input text.

        Args:
            text (str): The input text to be processed.
        Returns:
            str: The processed text.
        """
        return text, True, {}

    def apply_pairs(self, text1: str, text2: str, **kwargs) -> tuple[str, str]:
        """
        Process two input texts.

        Args:
            text1 (str): The first input text to be processed.
            text2 (str): The second input text to be processed.
        Returns:
            tuple[str, str]: A tuple containing the processed texts.
        """
        return (text1, text2), True, {}