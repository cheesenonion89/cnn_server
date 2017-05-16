import unittest
from unittest import TestCase

import os

import slim.train_root_models as trainer
from cnn_server.server import file_service as dirs

"""
class TestTrainRootModels(TestCase):
    def test_train_root_model(self):
        root_model_protobufs = dirs.get_protobuf_dir('root')

        if not os.path.exists(root_model_protobufs) or not os.listdir(root_model_protobufs):
            print('No data for root model training in %s' % root_model_protobufs)
            return None

        pretrained_model_name = 'inception_v4_pretrained'
        untrained_model_name = 'vgg_19'

        trainer.train_root_model(pretrained_model_name, max_train_time=100)
        pretrained_model_ckpt_path = dirs.get_root_model_dir(pretrained_model_name)
        self.assertTrue(os.path.exists(pretrained_model_ckpt_path))
        self.assertTrue(os.listdir(pretrained_model_ckpt_path))

        # Clean the root model directory for next test run
        for file in os.listdir(pretrained_model_ckpt_path):
            file_path = os.path.join(pretrained_model_ckpt_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        trainer.train_root_model(untrained_model_name, max_train_time=100)
        untrained_model_ckpt_path = dirs.get_root_model_dir(untrained_model_name)
        self.assertTrue(os.path.exists(untrained_model_ckpt_path))
        self.assertTrue(os.listdir(untrained_model_ckpt_path))

        # Clean the root model directory for next test run
        for file in os.listdir(untrained_model_ckpt_path):
            file_path = os.path.join(untrained_model_ckpt_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    unittest.main()
"""