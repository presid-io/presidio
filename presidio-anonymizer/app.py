"""TODO: Flask application instantiation and route definition."""

from flask import Flask, request

app = Flask(__name__)


# TODO add method for get supported anonymizers. 2629
@app.route("/anonymize", methods=["POST"])
def anonymize():
    """
    TODO: Anonymize endpoint definition.

    :return:
    """
    content = request.json
    return content


if __name__ == "main":
    app.run()