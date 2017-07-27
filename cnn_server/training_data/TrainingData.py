from flask import request
from flask_restful import Resource

from cnn_server.training_data import training_data_receive_handler as handler

'''
HTTP Endpoints for the upload of training data
'''


class TrainingData(Resource):
    def put(self, bot_id: int):

        print("PUT RECEIVED FOR BOT %s" % bot_id)

        try:
            int(bot_id)
        except ValueError:
            print("Invalid bot ID format. Expected integer value")
            return "Invalid bot ID format. Expected integer value", 400

        training_data_file = request.files['file']

        if not training_data_file:
            print("Training Data File is missing")
            return "Training Data File is missing", 400

        print("GOT TRAINING DATA FOR BOT %s" % bot_id)

        return handler.handle_put(bot_id, training_data_file)

    def delete(self, bot_id):

        print("DELETE RECEIVED FOR BOT %s" % bot_id)

        try:
            int(bot_id)
        except ValueError:
            print("Invalid bot ID format. Expected integer value")
            return "Invalid bot ID format. Expected integer value", 400

        handler.handle_delete(bot_id)
