from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class UsPassportRecognizer(PatternRecognizer):
    """
    Recognizes US Passport number using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    # pylint: disable=line-too-long,abstract-method
    # Weak pattern: all passport numbers are a weak match, e.g., 14019033
    PATTERNS = [
        Pattern("Passport (very weak)", r"(\b[0-9]{9}\b)", 0.05),
    ]
    CONTEXT = ["us", "united", "states", "passport", "passport#", "travel", "document"]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "US_PASSPORT",
    ):
        context = context if context else self.CONTEXT
        patterns = patterns if patterns else self.PATTERNS
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )
