import unittest
from unittest import TestCase

import os
import shutil

from cnn_server.server import file_service as dirs
from cnn_server.training_data import training_data_service as service

FILES_DIR = 'files'
TRAINING_DATA_DIR = '/home/markus/projects/cnn_server/training_data/'
BOT_ID = 1
BOT_TRAINING_DATA_DIR = dirs.get_training_data_dir(BOT_ID)
BOT_PROTOBUF_DIR = dirs.get_protobuf_dir(BOT_ID)


class TestTrainingDataService(TestCase):
	def test_validate_training_data(self):

		# Read the ZIP Files
		valid_zip = os.path.join(FILES_DIR, 'valid_trainingdata.zip')
		invalid_zip_subfolder = os.path.join(FILES_DIR, 'invalid_training_data_subfolder.zip')
		invalid_zip_file = os.path.join(FILES_DIR, 'invalid_training_data_file.zip')
		invalid_zip_emptysub = os.path.join(FILES_DIR, 'invalid_training_data_emptysub.zip')
		invalid_zip_emptysubend = os.path.join(FILES_DIR, 'invalid_training_data_emptysubend.zip')
		invalid_flowers = os.path.join(FILES_DIR, 'invalid_flower_photos.zip')
		some_file_path = os.path.join(FILES_DIR, 'some_file.txt')

		self.assertFalse(service.validate_training_data(some_file_path))
		self.assertFalse(service.validate_training_data(invalid_zip_subfolder))
		self.assertFalse(service.validate_training_data(invalid_zip_file))
		self.assertFalse(service.validate_training_data(invalid_zip_emptysub))
		self.assertFalse(service.validate_training_data(invalid_zip_emptysubend))
		self.assertFalse(service.validate_training_data(invalid_flowers))
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

	def test_write_to_protobuffer(self):
		if os.path.exists(BOT_TRAINING_DATA_DIR):
			shutil.rmtree(BOT_TRAINING_DATA_DIR)

		service.create_training_data_dir(BOT_ID, os.path.join(FILES_DIR, 'flower_photos.zip'))
		service.write_to_protobuffer(BOT_ID, 'name')

		# Check if the bot directory has been created
		self.assertTrue(os.path.isdir(BOT_TRAINING_DATA_DIR))

		# Check if the labels.txt has been created
		self.assertTrue(os.path.isfile(os.path.join(BOT_PROTOBUF_DIR, 'labels.txt')))

		# Make sure the labels file contains as mainy files as the training data folder has subfolders
		with open(os.path.join(BOT_PROTOBUF_DIR, 'labels.txt')) as f:
			for lndx, dir in enumerate(os.listdir(BOT_TRAINING_DATA_DIR)):
				pass
			for fndx, ln in enumerate(f):
				pass
			self.assertEqual(lndx, fndx)

		# Make sure there are 10
		protofiles = 0
		training_files = 0
		validation_files = 0
		for file in os.listdir(BOT_PROTOBUF_DIR):
			if file.endswith('.tfrecord'):
				protofiles += 1
			if 'train' in file:
				training_files += 1
			if 'validation' in file:
				validation_files += 1
		self.assertEqual(10, protofiles)
		self.assertEqual(5, training_files)
		self.assertEqual(5, validation_files)

		if os.path.exists(BOT_PROTOBUF_DIR):
			shutil.rmtree(BOT_PROTOBUF_DIR)

		if os.path.exists(BOT_TRAINING_DATA_DIR):
			shutil.rmtree(BOT_TRAINING_DATA_DIR)

		self.assertTrue(True)


if __name__ == '__main__':
	unittest.main()
