from flask_restful import Resource

from cnn_server.transfer_learning import transfer_learning_receive_handler as handler

_DEFAULT_TRAIN_TIME = (60 * 5)  # 5 Minutes
_TEST = False

'''
HTTP endpoint to trigger the transfer learning
'''
class TransferLearning(Resource):
    def put(self, bot_id):
        return handler.handle_put(bot_id, test=_TEST, max_train_time=_DEFAULT_TRAIN_TIME)
