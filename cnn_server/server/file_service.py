import os

PROJECT_ROOT_DIR = '/home/markus/projects/cnn_server/'
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'training_data')
PROTOBUF_DIR = os.path.join(PROJECT_ROOT_DIR, 'protobuf')
MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, 'model')

FOLDER_PREFIX = 'bot'


def folder_name(bot_id):
	return '%s_%s/' % (FOLDER_PREFIX, bot_id)


def get_training_data_dir(bot_id):
	return os.path.join(TRAINING_DATA_DIR, folder_name(bot_id))


def get_protobuf_dir(bot_id):
	return os.path.join(PROTOBUF_DIR, folder_name(bot_id))


def get_model_data_dir(bot_id):
	return os.path.join(MODEL_DIR, folder_name(bot_id))


def get_root_model_dir():
	return os.path.join(MODEL_DIR, 'root')
