from abc import ABC
from typing import List

from presidio_analyzer import EntityRecognizer
from presidio_analyzer.nlp_engine import NlpArtifacts


class RemoteRecognizer(ABC, EntityRecognizer):
    """
    A configuration for a recognizer that runs on a different process / remote machine.

    :param supported_entities: A list of entities this recognizer can identify
    :param name: name of recognizer
    :param supported_language: The language this recognizer can detect entities in
    :param version: Version of this recognizer
    """

    def __init__(
        self,
        supported_entities: List[str],
        name: str,
        supported_language: str,
        version: str,
    ):
        super().__init__(supported_entities, name, supported_language, version)

    def load(self):  # noqa D102
        pass

    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
    ):  # noqa ANN201
        """
        Call an external service for PII detection.

        :param text: text to be analyzed
        :param entities: Entities that should be looked for
        :param nlp_artifacts: Additional metadata from the NLP engine
        :return: List of identified PII entities
        """

        # Add code here to connect to the side car
        pass

    def get_supported_entities(self) -> List[str]:  # noqa D102
        pass
