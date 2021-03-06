import unittest
from unittest import TestCase

import os

import slim.train_root_models as train
from cnn_server.server import file_service as dirs


class TestServer(TestCase):
    def test_get_root_model_dir(self):
        root_model_dir = dirs.get_root_model_dir(model_name='inception_v4')
        self.assertTrue(os.path.exists(root_model_dir))

        self.assertFalse(dirs.get_root_model_dir(model_name='hokuspokus'))

    def test_get_root_model_ckpt_path(self):
        model_pretrained = 'inception_v4_pretrained'
        model_not_pretrained = 'lenet'

        ckpt_path = dirs.get_root_model_ckpt_path(train.networks_map[model_pretrained])
        self.assertTrue(ckpt_path)

        ckpt_path_none = dirs.get_root_model_ckpt_path(train.networks_map[model_not_pretrained])
        self.assertFalse(ckpt_path_none)

    def test_get_datasets(self):
        train_dataset = dirs.get_root_model_training_file()
        self.assertTrue(os.path.isfile(train_dataset))

        bmw_model_file = dirs.get_transfer_learning_file('bmw_models')
        car_types_file = dirs.get_transfer_learning_file('car_types')
        cars_file = dirs.get_transfer_learning_file('cars')
        seasons_file = dirs.get_transfer_learning_file('seasons')

        self.assertTrue(os.path.isfile(bmw_model_file))
        self.assertTrue(os.path.isfile(car_types_file))
        self.assertTrue(os.path.isfile(cars_file))
        self.assertTrue(os.path.isfile(seasons_file))

    def test_get_bot_id_from_dir(self):
        bmw_models_bot_id = 'bmw_models'
        training_data_dir = dirs.get_training_data_dir(bmw_models_bot_id)
        protobuf_dir = dirs.get_protobuf_dir(bmw_models_bot_id)
        model_dir = dirs.get_model_data_dir(bmw_models_bot_id)
        bot_id = dirs.get_bot_id_from_dir(training_data_dir)
        self.assertEqual(bmw_models_bot_id, bot_id, 'bot ids do not match')
        bot_id = dirs.get_bot_id_from_dir(protobuf_dir)
        self.assertEqual(bmw_models_bot_id, bot_id, 'bot ids do not match')
        bot_id = dirs.get_bot_id_from_dir(model_dir)
        self.assertEqual(bmw_models_bot_id, bot_id, 'bot ids do not match')



if __name__ == '__main__':
    unittest.main()
