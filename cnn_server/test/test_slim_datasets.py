import unittest
from unittest import TestCase

import tensorflow.contrib.slim as tf_slim

import cnn_server.server.file_service as dirs
import slim.datasets.bot_dataset as data
import slim.datasets.dataset_factory as factory
import slim.datasets.dataset_utils as utils
import slim.datasets.download_training_data as td

BOT_ID = 'test'
BOT_PROTOBUF_DIR = dirs.get_protobuf_dir(BOT_ID)
BOT_TRAINING_DATA_DIR = dirs.get_training_data_dir(BOT_ID)
FILES_DIR = 'files'
NUMBER_OF_TEST_CLASSES = 12


class TestSlimDatasets(TestCase):
    def test_get_number_of_classes_by_labels(self):
        txt_labels = utils.get_number_of_classes_by_labels(BOT_PROTOBUF_DIR)
        self.assertEqual(5, txt_labels)
        self.assertNotEquals(6, txt_labels)

        sf_labels = utils.get_number_of_classes_by_subfolder(BOT_TRAINING_DATA_DIR)
        self.assertEqual(NUMBER_OF_TEST_CLASSES, sf_labels)
        self.assertNotEquals(6, sf_labels)

    def test_bot_dataset(self):
        train_set = data.get_split('train', BOT_PROTOBUF_DIR)
        validation_set = data.get_split('validation', BOT_PROTOBUF_DIR)
        self.assertTrue(train_set)
        self.assertTrue(type(train_set) is tf_slim.dataset.Dataset)
        self.assertEqual(train_set.num_classes, 5)
        self.assertEqual(train_set.num_samples, 3320)

        self.assertTrue(validation_set)
        self.assertTrue(type(validation_set) is tf_slim.dataset.Dataset)
        self.assertEqual(validation_set.num_classes, 5)
        self.assertEqual(validation_set.num_samples, 350)

    def test_dataset_factory(self):
        train_set = factory.get_dataset('bot', 'train', BOT_PROTOBUF_DIR)
        validation_set = factory.get_dataset('bot', 'validation', BOT_PROTOBUF_DIR)

        self.assertTrue(train_set)
        self.assertTrue(type(train_set) is tf_slim.dataset.Dataset)
        self.assertEqual(train_set.num_classes, 5)
        self.assertEqual(train_set.num_samples, 3320)

        self.assertTrue(validation_set)
        self.assertTrue(type(validation_set) is tf_slim.dataset.Dataset)
        self.assertEqual(validation_set.num_classes, 5)
        self.assertEqual(validation_set.num_samples, 350)

    def test_download_training_data(self):
        td.download_training_data('test', dirs.get_test_training_file())
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
