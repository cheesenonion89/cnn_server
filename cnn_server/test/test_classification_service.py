import unittest
from unittest import TestCase

import base64
import os
import tempfile

import cnn_server.classification.classification_service as service

FILES_DIR = 'files'
TEST_BOT_ID = 'test'


class TestClassification_service(TestCase):
	def test_classify_image(self):
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


if __name__ == '__main__':
	unittest.main()
