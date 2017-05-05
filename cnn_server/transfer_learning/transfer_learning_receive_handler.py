from cnn_server.transfer_learning import transfer_learning_service as service


def handle_put(bot_id: int):
	service.train(bot_id)
