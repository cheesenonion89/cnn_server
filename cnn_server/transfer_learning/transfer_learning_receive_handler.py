from cnn_server.transfer_learning import transfer_learning_service as service


def handle_put(bot_id: int):
	if service.train(bot_id):
		return 'Transfer Learning of Bot %s was successful.' % bot_id, 200
	else:
		return 'Error in Transfer Learning of Bot %s' % bot_id, 500
