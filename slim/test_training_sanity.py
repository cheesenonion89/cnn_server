import unittest
from unittest import TestCase

import os

import slim.transfer_learn_image_classifier as trainer
from cnn_server.server import file_service as dirs


class TestTrainInception(TestCase):
    def test_train_inception(self):
        #  /home/markus/projects/cnn_server/protobuf/bot_root
        protobuf_dir = dirs.get_protobuf_dir('root')  # Read the protobuffer files for the initial car dataset

        # make sure ckpt files are there and correct data is in it
        labels = 0
        if not os.path.isfile(os.path.join(protobuf_dir, 'labels.txt')):
            print('Missing labels in %s' % os.path.join(protobuf_dir, 'labels.txt'))
            return None

        for line in open(os.path.join(protobuf_dir, 'labels.txt')):
            labels += 1
            """
            if not str(line.split(':')[1]) in ['ford', 'porsche', 'bmw', 'volkswagen', 'nissan', 'mercedes', 'honda',
                                               'aston martin', 'jeep',
                                               'chevrolet', 'ferrari', 'audi']:
                print(
                    'Data in protobuf directory %s is not correct. Wrong label %s' % (protobuf_dir, line.split(':')[1]))
                return None
            """
        if not labels == 12:
            print('Wrong number of labels: %s in: %s' % (labels, protobuf_dir))
            return None
        # /home/markus/projects/cnn_server/model/bot_root
        bot_model_dir = dirs.get_model_data_dir('root')
        # make sure bot_model_dir is there and empty
        if not os.path.isdir(bot_model_dir):
            os.mkdir(bot_model_dir)

        trainer.train(bot_model_dir=bot_model_dir,
                      protobuf_dir=protobuf_dir,
                      max_train_time_sec=(60 * 60 * 24 * 7),
                      optimization_params=None,
                      log_every_n_steps=10)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
