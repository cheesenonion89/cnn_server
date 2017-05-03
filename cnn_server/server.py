
from cnn_server.classification.Classifier import Classifier
from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from cnn_server.server.Server import Server
from cnn_server.training_data.TrainingData import TrainingData
from cnn_server.transfer_learning.TransferLearning import TransferLearning

app = Flask(__name__)
CORS(app)
api = Api(app)

api.add_resource(TrainingData, '/training_data/<bot_id>')
api.add_resource(TransferLearning, '/train/<bot_id>')
api.add_resource(Classifier, '/classify/<bot_id>')

api.add_resource(Server, '/status')

if __name__ == "__main__":
	app.run('0.0.0.0')
