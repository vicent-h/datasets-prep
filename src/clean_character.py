import re
from src.processor import Processor

class CleanCharacter(Processor):
    def __init__(
            self,
            remove_tags: bool = True,
            remove_repeated_characters: bool = True,
            remove_repeated_words: bool = True,
            captialize_sentences: bool = True
        ):
        self.use_remove_tags = remove_tags
        self.use_remove_repeated_characters = remove_repeated_characters
        self.use_remove_repeated_words = remove_repeated_words
        self.use_captialize_sentences = captialize_sentences

    def apply(self, text: str) -> str:
        """
        Clean the input text based on the specified cleaning options.

        Args:
            text (str): The input text to be cleaned.
        Returns:
            str: The cleaned text.
        """
        if self.use_remove_tags:
            text, _ = self.remove_tags(text)
        if self.use_remove_repeated_characters:
            text, _ = self.remove_repeated_characters(text)
        if self.use_remove_repeated_words:
            text, _ = self.remove_repeated_words(text)
        if self.use_captialize_sentences:
            text, _ = self.captialize_sentences(text)

        return text, True, {}

    def apply_pairs(self, text1: str, text2: str, **kwargs) -> tuple[str, str]:
        """
        Clean two input texts based on the specified cleaning options.

        Args:
            text1 (str): The first input text to be cleaned.
            text2 (str): The second input text to be cleaned.
        Returns:
            tuple[str, str]: A tuple containing the cleaned texts.
        """
        if self.remove_tags:
            text1, _ = self.remove_tags(text1)
            text2, _ = self.remove_tags(text2)
        if self.remove_repeated_characters:
            text1, _ = self.remove_repeated_characters(text1)
            text2, _ = self.remove_repeated_characters(text2)
        if self.remove_repeated_words:
            text1, _ = self.remove_repeated_words(text1)
            text2, _ = self.remove_repeated_words(text2)
        if self.captialize_sentences:
            text1, _ = self.captialize_sentences(text1)
            text2, _ = self.captialize_sentences(text2)

        return (text1, text2), True, {}

    def remove_tags(self, text: str) -> str:
        """
        Remove HTML/XML tags from the input text.

        Args:
            text (str): The input text to be cleaned.
        Returns:
            str: The cleaned text without tags.
        """
        clean_text = re.sub(r'<[^>]+>', '', text)
        return clean_text, True

    def remove_repeated_characters(self, text: str) -> str:
        """
        Remove repeated characters from the input text.

        Args:
            text (str): The input text to be cleaned.
        Returns:
            str: The cleaned text with repeated characters removed.
        """

        clean_text = re.sub(r'(.)\1{3,}', r'\1', text)
        return clean_text, True

    def remove_repeated_words(self, text: str) -> str:
        """
        Remove repeated words from the input text.

        Args:
            text (str): The input text to be cleaned.
        Returns:
            str: The cleaned text with repeated words removed.
        """
        clean_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
        return clean_text, True

    def captialize_sentences(self, text: str) -> str:
        """
        Capitalize the first letter of each sentence in the input text.

        Args:
            text (str): The input text to be cleaned.
        Returns:
            str: The cleaned text with sentences capitalized.
        """
        clean_text = re.sub(r'(?<=[.!?])\s*(\w)', lambda m: m.group(1).upper(), text)
        return clean_text, True