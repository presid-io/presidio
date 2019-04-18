from unittest import TestCase

import os
import pytest

from analyzer.predefined_recognizers import CreditCardRecognizer, \
    UsPhoneRecognizer, DomainRecognizer, UsItinRecognizer, \
    UsLicenseRecognizer, UsBankRecognizer, UsPassportRecognizer, \
    IpRecognizer, UsSsnRecognizer
from analyzer.nlp_engine import SpacyNlpEngine, NlpArtifacts

ip_recognizer = IpRecognizer()
us_ssn_recognizer = UsSsnRecognizer()
phone_recognizer = UsPhoneRecognizer()
us_itin_recognizer = UsItinRecognizer()
us_license_recognizer = UsLicenseRecognizer()
us_bank_recognizer = UsBankRecognizer()
us_passport_recognizer = UsPassportRecognizer()

@pytest.fixture(scope="class")
def sentences_with_context(request):
    """ Loads up a group of sentences with relevant context words
    """

    path = os.path.dirname(__file__) + '/data/context_sentences_tests.txt'
    f = open(path, "r")
    if not f.mode == 'r':
        return []
    content = f.read()
    f.close()

    lines = content.split('\n')
    # remove empty lines
    lines = list(filter(lambda k: k.strip(), lines))
    # remove comments
    lines = list(filter(lambda k: k[0] != '#', lines))

    test_items = []
    for i in range(len(lines)):
        if i % 2 == 1:
            continue
        recognizer = None
        entity_type = lines[i].strip()
        if entity_type == "IP_ADDRESS":
            recognizer = ip_recognizer
        elif entity_type == "US_SSN":
            recognizer = us_ssn_recognizer
        elif entity_type == "PHONE_NUMBER":
            recognizer = phone_recognizer
        elif entity_type == "US_ITIN":
            recognizer = us_itin_recognizer
        elif entity_type == "US_DRIVER_LICENSE":
            recognizer = us_license_recognizer
        elif entity_type == "US_BANK_NUMBER":
            recognizer = us_bank_recognizer
        elif entity_type == "US_PASSPORT":
            recognizer = us_passport_recognizer
        else:
            # will fail the test in its turn
            print("bad type: ", entity_type)
            return []
        test_items.append((lines[i+1].strip(),
                           recognizer,
                           [lines[i].strip()]))
    request.cls.context_sentences = test_items

@pytest.mark.usefixtures("sentences_with_context")
class TestContextSupport(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestContextSupport, self).__init__(*args, **kwargs)

    # Context tests
    def test_text_with_context_improves_score(self):
        # Currently we have 19 sentences, this is a sanity
        assert(len(self.context_sentences) == 19)
        nlp_engine = SpacyNlpEngine()
        mock_nlp_artifacts = NlpArtifacts([], [], [], [], None, "en")

        for item in self.context_sentences:
            text = item[0]
            recognizer = item[1]
            entities = item[2]
            nlp_artifacts = nlp_engine.process_text(text, "en")
            results_without_context = recognizer.analyze(text, entities, mock_nlp_artifacts)
            results_with_context = recognizer.analyze(text, entities, nlp_artifacts)

            assert(len(results_without_context) == len(results_with_context))
            for i in range(len(results_with_context)):
                assert(results_without_context[i].score < results_with_context[i].score)
