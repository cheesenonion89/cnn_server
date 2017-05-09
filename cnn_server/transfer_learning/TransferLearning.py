from flask_restful import Resource

from cnn_server.transfer_learning import transfer_learning_receive_handler as handler


class TransferLearning(Resource):
	def get(self, bot_id):
		"""

		:param bot_id: 
		:return: 
		"""
		return True

	def put(self, bot_id):
		"""

		:param bot_id: 
		:return: 
		"""
		handler.handle_put(bot_id)

		return True
