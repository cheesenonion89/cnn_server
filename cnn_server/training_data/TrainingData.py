from flask import request
from flask_restful import Resource
from zipfile import ZipFile

from cnn_server.training_data import training_data_receive_handler as handler


class TrainingData(Resource):
	def put(self, bot_id: int):
		"""
		
		:param bot_id: 
		:return: 
		"""

		try:
			int(bot_id)
		except ValueError:
			return "Invalid bot ID format. Expected integer value", 400

		if not request.files['file']:
			return "Training Data File is missing", 400

		if not request.form['net']:
			return "CNN Identifier is missing", 400

		training_data_file = request.files['file']
		net = request.form['net']

		return handler.handle_put(bot_id, training_data_file, net)
