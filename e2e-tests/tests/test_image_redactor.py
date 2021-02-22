import os

import pytest

from common.methods import redact


@pytest.mark.api
def test_given_image_with_tuple_color_filling_then_we_get_redact_image():
    # purple redact
    expected_result_image = get_file("purple_redact.png")
    response = redact(get_file("original_image.png"), "102, 0, 102")
    validate_and_close(expected_result_image, response)


@pytest.mark.api
def test_given_no_image_then_we_fail():
    # black redact
    expected_response = '{"error": "Invalid parameter, please insert image data"}'
    response = redact("")
    assert response.status_code == 422
    assert response.content.decode() == expected_response


def get_file(file_name: str):
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "..", "resources", file_name)
    return open(file_path, "rb")


def validate_and_close(expected_result_image, response):
    assert response.status_code == 200
    assert response.content == expected_result_image.read()
    expected_result_image.close()
