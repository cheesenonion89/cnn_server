import unittest
from unittest import TestCase
from zipfile import ZipFile

import os
import shutil

from cnn_server.server import file_service as dirs
from cnn_server.training_data import training_data_service as service

FILES_DIR = 'files'
TRAINING_DATA_DIR = '/home/markus/projects/cnn_server/training_data/'
BOT_ID = 1
BOT_TRAINING_DATA_DIR = dirs.get_training_data_dir(BOT_ID)


class TestTrainingDataService(TestCase):
	def test_validate_training_data(self):

		# Read the ZIP Files
		valid_zip = os.path.join(FILES_DIR, 'valid_trainingdata.zip')
		invalid_zip_subfolder = os.path.join(FILES_DIR, 'invalid_training_data_subfolder.zip')
		invalid_zip_file = os.path.join(FILES_DIR, 'invalid_training_data_file.zip')
		invalid_zip_emptysub = os.path.join(FILES_DIR, 'invalid_training_data_emptysub.zip')
		invalid_zip_emptysubend = os.path.join(FILES_DIR, 'invalid_training_data_emptysubend.zip')
		some_file_path = os.path.join(FILES_DIR, 'some_file.txt')

		self.assertFalse(service.validate_training_data(some_file_path))
		self.assertFalse(service.validate_training_data(invalid_zip_subfolder))
		self.assertFalse(service.validate_training_data(invalid_zip_file))
		self.assertFalse(service.validate_training_data(invalid_zip_emptysub))
		self.assertFalse(service.validate_training_data(invalid_zip_emptysubend))
		self.assertTrue(service.validate_training_data(valid_zip))
		self.assertEqual(1, 1)


	def test_create_training_data_dir(self):
		# Clear the directory first
		if os.path.exists(BOT_TRAINING_DATA_DIR):
			shutil.rmtree(BOT_TRAINING_DATA_DIR)

		service.create_training_data_dir(BOT_ID, os.path.join(FILES_DIR, 'valid_trainingdata.zip'))

		self.assertTrue(os.listdir(TRAINING_DATA_DIR), 'No data in training data dir after extraction')
		self.assertTrue(os.listdir(BOT_TRAINING_DATA_DIR), 'No data in bot training data dir after extraction')

		n_dirs = 0
		for dir in os.listdir(BOT_TRAINING_DATA_DIR):
			n_dirs += 1
			self.assertEqual('class', dir[0:5], 'subfolder names are not as expected')

		self.assertEqual(2, n_dirs, 'wrong number of directories after file extraction')

		shutil.rmtree(BOT_TRAINING_DATA_DIR)


if __name__ == '__main__':
	unittest.main()
