import unittest
from unittest import TestCase

import base64
import os
import tempfile
import json

import cnn_server.classification.classification_receive_handler as handler

TEST_BOT_ID = 'test'
FILES_DIR = 'files'


class TestClassification_receive_handler(TestCase):
	def test_handle_post(self):

		expected_return_labels = 3

		temp_file = tempfile.NamedTemporaryFile()
		temp_file.write(
			base64.b64encode(
				open(
					os.path.join(FILES_DIR, 'tulip.jpg'), "rb"
				).read()
			)
		)
		temp_file.seek(0)

		json_result, status = handler.handle_post(TEST_BOT_ID, temp_file.read(), return_labels=expected_return_labels)

		temp_file.close()

		json_result = json.loads(json_result)
		labels = json_result['labels']
		probs = json_result['probabilities']
		self.assertTrue(labels)
		self.assertTrue(probs)
		self.assertEqual(expected_return_labels, len(labels))
		self.assertEqual(expected_return_labels, len(probs))


if __name__ == '__main__':
	unittest.main()
