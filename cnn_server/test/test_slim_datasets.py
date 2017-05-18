import unittest
from unittest import TestCase

import os
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
    def test_get_split_size(self):
        bmw_model_bot_id = 'bmw_models'
        bmw_model_bot_dir = dirs.get_training_data_dir(bmw_model_bot_id)
        if not bmw_model_bot_dir:
            print('Bot Training Data Dir %s is not available. Test cannot run.' % bmw_model_bot_dir)
            return None
        expected_train_set_size = 4099
        expected_val_set_size = 455
        train_split = utils.get_split_size(bmw_model_bot_dir, 'train')
        val_split = utils.get_split_size(bmw_model_bot_dir, 'validation')
        self.assertIn(expected_train_set_size - train_split, range(-1, 2))
        self.assertIn(expected_val_set_size - val_split, range(-1, 2))

    def test_get_number_of_classes(self):
        exp_nr_of_classes_lbl = 0
        with open(os.path.join(BOT_PROTOBUF_DIR, 'labels.txt')) as f:
            for ndx, ln in enumerate(f):
                pass
            exp_nr_of_classes_lbl = ndx + 1

        exp_nr_of_classes_subf = 0
        for ndx, dir in enumerate(os.listdir(BOT_TRAINING_DATA_DIR)):
            exp_nr_of_classes_subf = ndx
        exp_nr_of_classes_subf += 1

        if not exp_nr_of_classes_lbl == exp_nr_of_classes_subf:
            print("Test invalid. Expected values are not matching %s != %s" % (
                exp_nr_of_classes_lbl, exp_nr_of_classes_subf))
            return None

        txt_labels = utils.get_number_of_classes_by_labels(BOT_PROTOBUF_DIR)

        self.assertEqual(exp_nr_of_classes_lbl, txt_labels)
        self.assertEqual(exp_nr_of_classes_subf, txt_labels)

        sf_labels = utils.get_number_of_classes_by_subfolder(BOT_TRAINING_DATA_DIR)
        self.assertEqual(exp_nr_of_classes_subf, sf_labels)
        self.assertEqual(exp_nr_of_classes_lbl, sf_labels)

    def test_bot_dataset(self):

        exp_train_set_size = utils.get_split_size(BOT_TRAINING_DATA_DIR, 'train')
        exp_val_set_size = utils.get_split_size(BOT_TRAINING_DATA_DIR, 'validation')
        exp_num_class = utils.get_number_of_classes_by_labels(BOT_PROTOBUF_DIR)

        train_set = data.get_split('train', BOT_PROTOBUF_DIR)
        validation_set = data.get_split('validation', BOT_PROTOBUF_DIR)
        self.assertTrue(train_set)
        self.assertTrue(type(train_set) is tf_slim.dataset.Dataset)
        self.assertEqual(train_set.num_classes, exp_num_class)
        self.assertEqual(train_set.num_samples, exp_train_set_size)

        self.assertTrue(validation_set)
        self.assertTrue(type(validation_set) is tf_slim.dataset.Dataset)
        self.assertEqual(validation_set.num_classes, exp_num_class)
        self.assertEqual(validation_set.num_samples, exp_val_set_size)

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

        bmw_models_bot_id = 'bmw_models'
        bmw_model_protobuf = dirs.get_protobuf_dir(bmw_models_bot_id)
        train_set = factory.get_dataset('bot', 'train', bmw_model_protobuf)
        validation_set = factory.get_dataset('bot', 'validation', bmw_model_protobuf)
        exp_num_classes = utils.get_number_of_classes_by_labels(bmw_model_protobuf)

        exp_train_set_size = utils.get_split_size(bmw_models_bot_id, 'train')
        exp_val_set_size = utils.get_split_size(bmw_models_bot_id, 'validation')

        self.assertTrue(train_set)
        self.assertTrue(type(train_set) is tf_slim.dataset.Dataset)
        self.assertEqual(train_set.num_classes, exp_num_classes)
        self.assertEqual(train_set.num_samples, exp_train_set_size)

        self.assertTrue(validation_set)
        self.assertTrue(type(validation_set) is tf_slim.dataset.Dataset)
        self.assertEqual(validation_set.num_classes, exp_num_classes)
        self.assertEqual(validation_set.num_samples, exp_val_set_size)

    def test_download_training_data(self):
        td.download_training_data('test', dirs.get_test_training_file())
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
