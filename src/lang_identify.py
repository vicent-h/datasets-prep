import numpy as np
import fasttext
from fasttext.FastText import _FastText as FastTextClass


def _predict_compat(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
    def check(entry):
        if entry.find("\n") != -1:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        entry += "\n"
        return entry

    if type(text) == list:
        text = [check(entry) for entry in text]
        all_labels, all_probs = self.f.multilinePredict(
            text, k, threshold, on_unicode_error
        )
        return all_labels, all_probs

    text = check(text)
    predictions = self.f.predict(text, k, threshold, on_unicode_error)
    if predictions:
        probs, labels = zip(*predictions)
    else:
        probs, labels = ([], ())

    return labels, np.asarray(probs)

FastTextClass.predict = _predict_compat

class LangIdentifier:
    def __init__(self, model_path: str = None, threshold: float = 0.75, k=1):
        self.threshold = threshold
        self.k = k
        if model_path is None:
            model_path = "/media/alvarinho/dados/Models/lid.176.bin"
        self.model_path = model_path
        self.model = fasttext.load_model(model_path)

    def apply(self, text: str, expected_lang: str = None) -> tuple:
        """
        Predict the language of the input text.

        Args:
            text (str): The input text to be classified.
            k (int): The number of top predictions to return.
            threshold (float): The probability threshold for predictions.

        Returns:
            tuple: A tuple containing a list of predicted labels and an array of probabilities.
        """
        if expected_lang is not None:
            expected_lang = expected_lang.lower()
        else:
            raise ValueError("expected_lang must be provided for language identification.")
        predict = self.model.predict(text, k=self.k)
        if self.k == 1:
            label, prob = predict
            lang = label[0].replace("__label__", "")
            if prob[0] > self.threshold and lang == expected_lang:
                return text, True
        else:
            for label, prob in zip(*predict):
                lang = label.replace("__label__", "")
                if prob > self.threshold and lang == expected_lang:
                    return text, True
        return text, False

    def score(self, text: str) -> float:
        """
        Get the confidence score of the language prediction for the input text.

        Args:
            text (str): The input text to be classified.
        
        Returns:
            float: The confidence score of the top prediction.
        """
        predict = self.model.predict(text, k=self.k)
        return predict[1][0]


if __name__ == "__main__":
    lang_identifier = LangIdentifier(k=2)

    examples_texts = [
        "This is an example sentence in English.",
        "Ceci est une phrase d'exemple en français.",
        "Dies ist ein Beispielsatz auf Deutsch.",
        "Esta es una oración de ejemplo en español.",
        "Essa é uma frase de exemplo em português.",
        "これは日本語の例文です。",
        "이것은 한국어 예문입니다.",
        "Это пример предложения на русском языке.",
        "هذا مثال لجملة باللغة العربية.",
        "यह हिंदी में एक उदाहरण वाक्य है।",
        "นี่คือตัวอย่างประโยคในภาษาไทย",
    ]

    for text in examples_texts:
        prediction = lang_identifier.score(text)
        print(text, prediction)

    print('Testando com expected_lang')

    test_cases = [
        ("This is an example sentence in English.", "en"),
        ("Ceci est une phrase d'exemple en français.", "fr"),
        ("Dies ist ein Beispielsatz auf Deutsch.", "de"),
        ("Esta es una oración de ejemplo en español.", "es"),
        ("Essa é uma frase de exemplo em português.", "pt")
    ]

    for text, expected_lang in test_cases:
        prediction = lang_identifier.apply(text, expected_lang=expected_lang)
        print(text, expected_lang, prediction)


