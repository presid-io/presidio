from pattern_recognizer import PatternRecognizer
from pattern import Pattern


class CreditCardRecognizer(PatternRecognizer):

    def __init__(self):
        patterns = []
        r = r'\b((4\d{3})|(5[0-5]\d{2})|(6\d{3})|(1\d{3})|(3\d{3}))[- ]?(\d{3,4})[- ]?(\d{3,4})[- ]?(\d{3,5})\b'  # noqa: E501
        p = Pattern('All Credit Cards (weak)', 0.3, r)
        patterns.append(p)

        context = [
          "credit",
          "card",
          "visa",
          "mastercard",
          # "american express" #TODO: add after adding keyphrase support
          "amex",
          "discover",
          "jcb",
          "diners",
          "maestro",
          "instapayment"
        ]

        super().__init__(["CREDIT_CARD"], [], patterns, None, context)

    def validate_pattern_logic(self, text, result):
        self.__sanitize_value(text)
        res = self.__luhn_checksum()
        if res == 0:
            result.score = 1
        else:
            result.score = 0

        return result

    def __luhn_checksum(self):
        def digits_of(n):
            return [int(d) for d in str(n)]

        digits = digits_of(self.sanitized_value)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = 0
        checksum += sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        return checksum % 10

    def __sanitize_value(self, text):
        self.sanitized_value = text.replace('-', '').replace(' ', '')
