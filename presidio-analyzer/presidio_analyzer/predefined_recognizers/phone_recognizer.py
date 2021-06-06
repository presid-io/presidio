from typing import List

import phonenumbers
from phonenumbers import COUNTRY_CODE_TO_REGION_CODE
from phonenumbers.geocoder import country_name_for_number

from presidio_analyzer import RecognizerResult, LocalRecognizer, AnalysisExplanation
from presidio_analyzer.nlp_engine import NlpArtifacts


SCORE = 0.6
ENTITY_TYPE_SUFFIX = "_PHONE_NUMBER"
INTERNATIONAL_ENTITY_TYPE = "INTERNATIONAL_PHONE_NUMBER"


class PhoneRecognizer(LocalRecognizer):
    """Recognize multi-regional phone numbers.

     Using python-phonenumbers, along with fixed and regional context words.
    :param supported_language: Language this recognizer supports
    :param supported_entities: The entities this recognizer can detect
    """

    CONTEXT = ["phone", "number", "telephone", "cell", "cellphone", "mobile", "call"]

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: List[str] = None,
    ):
        supported_entities = (
            self.get_supported_entities()
            if not supported_entities
            else supported_entities
        )
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
        )

    def load(self) -> None:  # noqa D102
        pass

    def get_supported_entities(self):  # noqa D102
        return [
            value[0] + ENTITY_TYPE_SUFFIX
            for value in COUNTRY_CODE_TO_REGION_CODE.values()
        ] + [INTERNATIONAL_ENTITY_TYPE]

    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None
    ) -> List[RecognizerResult]:
        """Analyzes text to detect phone numbers using python-phonenumbers.

        Iterates over entities, fetching regions, then matching regional
        phone numbers patterns against the text.
        :param text: Text to be analyzed
        :param entities: Entities this recognizer can detect
        :param nlp_artifacts: Additional metadata from the NLP engine
        :return: List of phone numbers RecognizerResults
        """
        results = []
        for entity in entities:
            region = entity.replace(ENTITY_TYPE_SUFFIX, "")
            for match in phonenumbers.PhoneNumberMatcher(text, region, leniency=0):
                international_phone_prefix = match.raw_string.startswith("+")
                if entity == INTERNATIONAL_ENTITY_TYPE and international_phone_prefix:
                    results += [self._get_international_recognizer_result(match)]
                # phone-numbers matches international numbers twice
                elif not international_phone_prefix:
                    results += [
                        self._get_regional_recognizer_result(
                            match, entity, text, nlp_artifacts
                        )
                    ]

        return results

    def _get_regional_recognizer_result(self, match, entity, text, nlp_artifacts):
        number = match.number
        main_region_code = COUNTRY_CODE_TO_REGION_CODE.get(number.country_code)[0]
        result = RecognizerResult(
            entity_type=entity,
            start=match.start,
            end=match.end,
            score=SCORE,
            analysis_explanation=PhoneRecognizer._get_analysis_explanation(),
        )
        # Enhance confidence using 'phone' related context and region code and name.
        region_specific_context = (
            self.CONTEXT
            + [main_region_code]
            + [country_name_for_number(number, self.supported_language)]
        )
        return self.enhance_using_context(
            text, [result], nlp_artifacts, region_specific_context
        )[0]

    @staticmethod
    def _get_international_recognizer_result(match):
        return RecognizerResult(
            entity_type=INTERNATIONAL_ENTITY_TYPE,
            start=match.start,
            end=match.end,
            score=0.6,
            analysis_explanation=PhoneRecognizer._get_analysis_explanation(),
        )

    @staticmethod
    def _get_analysis_explanation():
        return AnalysisExplanation(
            recognizer=PhoneRecognizer.__class__.__name__,
            original_score=SCORE,
            textual_explanation="Recognized using PhoneRecognizer",
        )
