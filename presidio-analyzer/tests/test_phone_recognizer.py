import pytest

from presidio_analyzer.predefined_recognizers.phone_recognizer import PhoneRecognizer
from tests import assert_result


@pytest.fixture(scope="module")
def recognizer():
    return PhoneRecognizer()


# Generate random address https://www.bitaddress.org/
@pytest.mark.parametrize(
    "text, expected_len, entities, expected_positions",
    [
        # fmt: off
        ("My US number is (415) 555-0132, and my international one is +1 415 555 0132", 2,
         ["INTERNATIONAL_PHONE_NUMBER", "US_PHONE_NUMBER"], ((60, 75), (16, 30),), ),
        ("My Israeli number is 09-7625400", 0, ["INTERNATIONAL_PHONE_NUMBER", "US_PHONE_NUMBER"], ()),
        ("My Israeli number is 09-7625400", 1, ["IL_PHONE_NUMBER"], ((21, 31), ), ),
        ("My Israeli number is 09-7625400", 125, PhoneRecognizer().get_supported_entities(), (), ),
        # fmt: on
    ],
)
def test_when_all_cryptos_then_succeed(
    text, expected_len, entities, expected_positions, recognizer,
):
    results = recognizer.analyze(text, entities)
    assert len(results) == expected_len
    for i, (res, (st_pos, fn_pos)) in enumerate(zip(results, expected_positions)):
        assert_result(res, entities[i], st_pos, fn_pos, 0.8)
