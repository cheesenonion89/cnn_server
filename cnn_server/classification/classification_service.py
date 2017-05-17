import base64
import os
import tempfile

import slim.inference_image_classifier as classifier


def classify_image(bot_id, image, return_labels=1):
    """
    Decodes the byte representation of a base64 encoded image and passes the result to the image classifier, together 
    with the bot id and the number of labels to return
    :param bot_id: the id of the bot, the CNN belongs to
    :param image: the image to be classified as base64 encoded byte representation
    :param return_labels: number of labels to return
    :return: returns the top n labels with probabilities, where n = return_labels
    """
    # Decode the image
    temp_file = tempfile.NamedTemporaryFile()
    temp_file.write(
        base64.b64decode(image)
    )

    labels, probabilities = classifier.inference_on_image(bot_id, os.path.join(tempfile.gettempdir(), temp_file.name),
                                                          network_name='inception_v3', return_labels=return_labels)

    temp_file.close()

    return labels, probabilities
