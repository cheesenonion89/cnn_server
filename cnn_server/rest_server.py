import time
import zipfile

import io
import os
import shutil
from flask import Flask, request, send_file
from flask_cors import CORS
from flask_restful import Api, Resource

TRAINING_DATA_DIR = 'training_data'
MODEL_DIR = 'models'

app = Flask(__name__)
CORS(app)
api = Api(app)


def get_training_data_dir(bot_id):
	return os.path.join(TRAINING_DATA_DIR, 'bot_%s' % bot_id)


def get_model_dir(bot_id):
	return os.path.join(MODEL_DIR, 'bot_%s' % bot_id)


def training_data_valid(bot_id):
	"""
	Checks if the training data directory for bot_id is in the format:
		training_data/
			bot_<bot_id>/
				<class_name_01>/
					<image_list>
				<class_name_02>/
					<image_list>
				...
	:param 
		bot_id: identifier of the bot
	:return: 
		True, if data is there and in the right format. False otherwise	
	"""
	return True


class TrainingData(Resource):
	def put(self, bot_id):
		"""
		Receives a zipped training data file, unzips and saves it at the training data folder for bot_id.
		If the data contained in the zip file is not in the right format, HTTP Status code 400 is returned
		:param 
			bot_id: identifier of the bot, the training data is uploaded for
		:return: 
			Status code indicating success of failure of save process
		"""
		print("TRAINING DATA RECEIVED")
		# bot_id = request.form['id']
		bot_training_dir = get_training_data_dir(bot_id)
		if not os.path.exists(bot_training_dir):
			os.mkdir(bot_training_dir)
		if os.listdir(bot_training_dir):
			return "training data already there", 200
		file = request.files['file']
		zipfile.ZipFile(file).extractall(bot_training_dir)

		# TODO: Trigger preprocessing and protobuf conversion to /slim/dataset/training_data_bot_id

		# TODO: Register training data at dataset factory with bot_id

		return "Successfully uploaded training data for bot %s" % bot_id, 201

	def get(self, bot_id):
		"""
		Returns the currently uploaded version of the training data to the client as ZIP file
		:param 
			bot_id: training data for the bot with the identifier bot_id
		:return: 
			zip file with the training data, if available, else HTTP Status 404
		"""
		print("Packing and Sending Traning Data for Bot %s" % bot_id)
		bot_training_dir = get_training_data_dir(bot_id)
		if not os.path.exists(bot_training_dir):
			return "there is no training data available for bot %s" % bot_id, 200
		temp_file = io.BytesIO()
		with zipfile.ZipFile(temp_file, 'w') as zip_file:
			for root, dirs, files in os.walk(bot_training_dir):
				for file in files:
					data = zipfile.ZipInfo(os.path.basename(file))
					data.date_time = time.localtime(time.time())[:6]
					data.compress_type = zipfile.ZIP_DEFLATED
					zip_file.writestr(data, file)
		temp_file.seek(0)
		return send_file(temp_file, attachment_filename='data.zip', as_attachment=True), 200

	def post(self, bot_id):
		"""
		Triggers the Transfer Learning based on the training data resource uploaded under bot_id.
		Fails if no training data is available or the training data is in the wrong format
		:param 
			bot_id: identifier of the bot to create an extended Inception CNN for
		:return: 
			HTTP Status message depending of success or failure of the process 
		"""
		if not training_data_valid(bot_id):
			return "The training data is not in the right format", 400

		training_data_dir = get_training_data_dir(bot_id)
		model_dir = get_model_dir(bot_id)

		# TODO: Trigger Transfer Learning and asynchronous response in POST
		return "Transfer Learning is started", 202

	def delete(self, bot_id):
		"""
		Deletes the training data and if availbale the transfer learning model for the bot with identifier bot_id 
		If no data is available a 404 will be returned
		:param 
			bot_id: identifier of the bot to delete the training data for
		:return: 
			HTTP Status message depending on the availability of the training data to be deleteed
		"""
		print("DELETING BOT %s" % bot_id)
		bot_training_dir = get_training_data_dir(bot_id)
		if not os.path.exists(bot_training_dir):
			return "no training data to delete for bot %s" % bot_id, 404
		shutil.rmtree(bot_training_dir)

		# TODO: Check if TransferLearning-Model is there, if so delete it too

		return "Deleted training data for bot %s" % bot_id, 200


class TransferLearning(Resource):
	def post(self, bot_id):
		return 'success', 200

	def get(self, bot_id):
		return "[imagenet]", 200


class Classifier(Resource):
	def post(self, bot_id):
		"""
		Accepts an image for classification and returns the classification result based on the transfer learning model
		Requires training data and transfer learning model to be available for bot_id
		:param 
			bot_id: identifier of the bot, the classification is intended to be done for
		:return: 
			Classification result and HTTP Status indicating success of failure
		"""
		return 'success', 200

	def put(self, bot_id):
		"""
		Maybe use to change model parameters. Later...
		:param bot_id: 
		:return: 
		"""
		return 'Parameter manipulation os currently not allowed', 423

	def get(self, bot_id):
		"""
		Maybe get information about model parameters. Later...
		:param bot_id: 
		:return: 
		"""
		return 'Parameter view currently not allowed', 423

	def delete(self, bot_id):
		"""
		Maybe delete the current model. Sort out responsibility conflict with TrainingData Resource then
		:param bot_id: 
		:return: 
		"""
		return 'Deletion of Training Data and Model via /training_data/<bot_id>', 423



class Status(Resource):
	# TODO: Return a meaningful status message
	def get(self):
		return 'online', 200


api.add_resource(TrainingData, '/training_data/<bot_id>')
api.add_resource(Status, '/status')

if __name__ == "__main__":
	app.run('0.0.0.0')
