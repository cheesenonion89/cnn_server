import unittest
from unittest import TestCase

import base64
import os
import shutil
import tempfile

import cnn_server.classification.classification_service as service
from cnn_server.server import file_service as dirs

FILES_DIR = 'files'
TEST_BOT_ID = 'test'


class TestClassificationService(TestCase):
	def test_classify_image(self):
		if os.path.exists(dirs.get_model_data_dir(TEST_BOT_ID)):
			shutil.rmtree(dirs.get_model_data_dir(TEST_BOT_ID))
		shutil.copytree(os.path.join(FILES_DIR, 'protobuf/bot_test'), dirs.get_model_data_dir(TEST_BOT_ID))

		temp_file = tempfile.NamedTemporaryFile()
		temp_file.write(
			base64.b64encode(
				open(
					os.path.join(FILES_DIR, 'daisy.jpg'), "rb"
				).read()
			)
		)
		temp_file.seek(0)

		labels, probabilities = service.classify_image(TEST_BOT_ID, temp_file.read())

		temp_file.close()
		self.assertEqual(1, len(labels))
		self.assertEqual(1, len(probabilities))
		# Clean the bot_model directory for next test run
		for file in os.listdir(dirs.get_model_data_dir(TEST_BOT_ID)):
			file_path = os.path.join(dirs.get_model_data_dir(TEST_BOT_ID), file)
			try:
				if os.path.isfile(file_path):
					os.unlink(file_path)
			except Exception as e:
				print(e)

if __name__ == '__main__':
	unittest.main()
