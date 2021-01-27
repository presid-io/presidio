"""Replaces the PII text entity with empty string."""
from presidio_anonymizer.anonymizers import Anonymizer


# TODO implement + test
class Redact(Anonymizer):
    """Redact the string - empty value."""

    def anonymize(self, original_text=None, params={}):
        """:return: an empty value."""
        pass
