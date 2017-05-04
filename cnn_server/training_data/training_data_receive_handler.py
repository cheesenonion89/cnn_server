from cnn_server.training_data import training_data_service


def handle_put(bot_id: int, training_data_file: str):
	"""
	
	:param bot_id: 
	:param training_data_file: 
	:param net: 
	:return: 
	"""

	# TODO: check if request_body is valid and redirect to training_data_service

	if not training_data_service.validate_training_data(training_data_file):
		return "Training Data is invalid", 400

	if not training_data_service.create_training_data_dir(bot_id, training_data_file):
		return "Failed to create training data directory.", 400

	if not training_data_service.write_to_protobuffer(bot_id):
		return "Failed to convert training data to protobuffer format", 400

	return "Training Data created", 200
