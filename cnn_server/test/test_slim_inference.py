import unittest
from unittest import TestCase

import numpy as np
import os
import shutil
import tempfile

import slim.inference_image_classifier as classifier
from cnn_server.server import file_service as dirs

TEST_BOT_ID = 'test'
FILES_DIR = 'files'


class TestInference(TestCase):
	def test_map_predictions_to_labels(self):
		mock_predictions = np.array([0.194, 0.534, 0.101, 0.022, 0.111])
		expected_return_labels = 3
		labels, probabilities = classifier.map_predictions_to_labels(TEST_BOT_ID, mock_predictions,
																	 expected_return_labels)

		self.assertEqual(len(labels), len(probabilities), "Labels als probabilities are unequally long")
		self.assertEqual(expected_return_labels, len(labels), "List of labels is not as long as expected")
		self.assertEqual(expected_return_labels, len(probabilities), "List of labels is not as long as expected")

		self.assertListEqual([0.534, 0.194, 0.111], probabilities)
		self.assertListEqual(['dandelion', 'daisy', 'tulips'], labels)

	# TODO: Restore CNN from a working ckpt file and also verify classification results
	def test_inference_on_image(self):
		if os.path.exists(dirs.get_model_data_dir(TEST_BOT_ID)):
			shutil.rmtree(dirs.get_model_data_dir(TEST_BOT_ID))
		shutil.copytree(os.path.join(FILES_DIR, 'protobuf/bot_test'), dirs.get_model_data_dir(TEST_BOT_ID))

		temp_file = tempfile.NamedTemporaryFile()
		temp_file.write(
			open(
				os.path.join(FILES_DIR, 'daisy.jpg'), "rb"
			).read()
		)

		labels, probabilities = classifier.inference_on_image(TEST_BOT_ID,
															  os.path.join(tempfile.gettempdir(), temp_file.name),
															  return_labels=1)

		temp_file.close()

		self.assertEqual(len(labels), len(probabilities))
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
