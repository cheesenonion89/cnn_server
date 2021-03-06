from flask import request
from flask_restful import Resource

import cnn_server.classification.classification_receive_handler as handler


class Classifier(Resource):
    def post(self, bot_id):
        """
        Provides a REST Interface to retrieve the classification of an image for a pretrained model belonging to a bot.
        The bot_id is passed as URL parameter /bot_id.
        The image is mandatory and needs to be part of the http Request Body as file, retrievable with the key 'image'.
        Optionally the number of return_labels can be passed to the Request Body Form with the key 'return_labels'
        :param bot_id: Id of the Bot to identify the correct Model
        :return: HTTP Response with the classification result or an Error
        """
        response = request.get_json(force=True)
        base64_image = response['base64Image']

        return handler.handle_post(bot_id, base64_image)
