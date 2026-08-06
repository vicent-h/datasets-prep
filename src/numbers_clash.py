import re

class NumbersClash:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def identify_numbers(text: str) -> list:
        """
        Identify numbers in the input text.

        Args:
            text (str): The input text to be analyzed.
        Returns:
            list: A list of numbers found in the text.
        """

        # Use regex to find numbers in the text
        numbers = re.findall(r'\d+', text)
        return numbers

    def apply(self, text1: str, text2: str) -> tuple:
        """
        Check if the two input texts have any numbers in common.

        Args:
            text1 (str): The first input text.
            text2 (str): The second input text.
        Returns:
            bool: True if there are common numbers, False otherwise.
        """
        numbers1 = self.identify_numbers(text1)
        numbers2 = self.identify_numbers(text2)

        # Check for common numbers
        common_numbers = set(numbers1).intersection(set(numbers2))
        if len(common_numbers) / max(len(numbers1), len(numbers2)) > self.threshold:
            return (text1, text2), True
        return (text1, text2), False