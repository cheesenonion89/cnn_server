import unittest
from unittest import TestCase

import os
import shutil

import cnn_server.server.file_service as dirs
import cnn_server.training_data.training_data_receive_handler as handler

FILES_DIR = 'files'
TRAINING_DATA_DIR = '/home/markus/projects/cnn_server/training_data/'
BOT_ID = 1
BOT_TRAINING_DATA_DIR = dirs.get_training_data_dir(BOT_ID)
BOT_PROTOBUF_DIR = dirs.get_protobuf_dir(BOT_ID)


class TestTraining_data_receive_handler(TestCase):
	def test_handle_put(self):
		if os.path.exists(BOT_PROTOBUF_DIR):
			shutil.rmtree(BOT_PROTOBUF_DIR)

		if os.path.exists(BOT_TRAINING_DATA_DIR):
			shutil.rmtree(BOT_TRAINING_DATA_DIR)

		vld_result, vld_status = handler.handle_put(BOT_ID, os.path.join(FILES_DIR, 'valid_trainingdata.zip'))
		self.assertEqual("Training Data created", vld_result)
		self.assertEqual(200, vld_status)

		self.assertEqual("Training Data is invalid",
						 handler.handle_put(BOT_ID, os.path.join(FILES_DIR, 'invalid_training_data_subfolder.zip'))[0])

		if os.path.exists(BOT_PROTOBUF_DIR):
			shutil.rmtree(BOT_PROTOBUF_DIR)

		if os.path.exists(BOT_TRAINING_DATA_DIR):
			shutil.rmtree(BOT_TRAINING_DATA_DIR)


if __name__ == '__main__':
	unittest.main()
