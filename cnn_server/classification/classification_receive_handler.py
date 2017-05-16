import json

import cnn_server.classification.classification_service as service


class ClassificationResult:
    def __init__(self, labels, probabilities):
        self.labels = labels
        self.probabilities = probabilities

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def handle_post(bot_id, image, return_labels=1):
    """
    Expects the bot id and the byte representation of a base64 encoded image, parsed from the incoming http POST
    :param bot_id: the id of the bot, whose CNN is to be referenced
    :param image: the image to be classified by the CNN
    :param return_labels: the number of result labels to return.
    :return: the top n classification results as JSON object to send back to the requesting party, where n = return_labels
    """
    labels, probabilities = service.classify_image(bot_id, image, return_labels)

    if not labels or not probabilities:
        return "Error processing the input image", 400
    elif len(labels) == 0 or len(probabilities) == 0:
        return "Classification Result is empty", 500
    else:
        return ClassificationResult(labels, probabilities).to_json(), 200
