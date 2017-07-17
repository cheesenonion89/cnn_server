from flask import request
from flask_restful import Resource

from cnn_server.training_data import training_data_receive_handler as handler


class TrainingData(Resource):
    def put(self, bot_id: int):
        """
        
        :param bot_id: 
        :return: 
        """
        print("PUT RECEIVED")
        try:
            int(bot_id)
        except ValueError:
            print("Invalid bot ID format. Expected integer value")
            return "Invalid bot ID format. Expected integer value", 400
        """
        if not request.files['file']:
            return "Training Data File is missing", 400

        if not request.form['net']:
            return "CNN Identifier is missing", 400

        
        # net = request.form['net']
        """
        r = request
        print(request)
        training_data_file = request.files['file']
        print("GOT TRAINING DATA")
        if not training_data_file:
            return "Training Data File is missing", 400

        return handler.handle_put(bot_id, training_data_file)
