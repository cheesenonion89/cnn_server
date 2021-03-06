import unittest
from unittest import TestCase

import os
import shutil

from cnn_server.server import file_service as dirs
from slim.datasets import convert_to_protobuf as converter

BOT_ID = 'bmw_models'


class TestConvertToProtobuf(TestCase):
    def test_get_filenames_and_classes(self):

        bmw3_exp = {'train': 2048, 'validation': 228}
        bmw5_exp = {'train': 515, 'validation': 57}
        bmw6_exp = {'train': 487, 'validation': 54}
        bmw7_exp = {'train': 1049, 'validation': 117}

        protobuf_dir = dirs.get_protobuf_dir(BOT_ID)
        training_data_dir = dirs.get_training_data_dir(BOT_ID)
        if not os.listdir(training_data_dir):
            print("Cannot start test. No data in %s" & training_data_dir)
            return
        if not os.path.exists(protobuf_dir):
            os.mkdir(protobuf_dir)
        if os.listdir(protobuf_dir):
            shutil.rmtree(protobuf_dir)
            os.mkdir(protobuf_dir)

        train, val, classes = converter._get_filenames_and_classes(training_data_dir, 0.1)
        bmw3_ctr_tr = 0
        bmw5_ctr_tr = 0
        bmw6_ctr_tr = 0
        bmw7_ctr_tr = 0
        for file in train:
            cl = os.path.basename(os.path.dirname(file))
            if cl == 'bmw3':
                bmw3_ctr_tr += 1
            elif cl == 'bmw5':
                bmw5_ctr_tr += 1
            elif cl == 'bmw6':
                bmw6_ctr_tr += 1
            elif cl == 'bmw7':
                bmw7_ctr_tr += 1

        bmw3_ctr_vl = 0
        bmw5_ctr_vl = 0
        bmw6_ctr_vl = 0
        bmw7_ctr_vl = 0
        for file in val:
            cl = os.path.basename(os.path.dirname(file))
            if cl == 'bmw3':
                bmw3_ctr_vl += 1
            elif cl == 'bmw5':
                bmw5_ctr_vl += 1
            elif cl == 'bmw6':
                bmw6_ctr_vl += 1
            elif cl == 'bmw7':
                bmw7_ctr_vl += 1
        self.assertIn(bmw3_exp['train'] - bmw3_ctr_tr, range(-2, 3))
        self.assertIn(bmw5_exp['train'] - bmw5_ctr_tr, range(-2, 3))
        self.assertIn(bmw6_exp['train'] - bmw6_ctr_tr, range(-2, 3))
        self.assertIn(bmw7_exp['train'] - bmw7_ctr_tr, range(-2, 3))
        self.assertIn(bmw3_exp['validation'] - bmw3_ctr_vl, range(-2, 3))
        self.assertIn(bmw5_exp['validation'] - bmw5_ctr_vl, range(-2, 3))
        self.assertIn(bmw6_exp['validation'] - bmw6_ctr_vl, range(-2, 3))
        self.assertIn(bmw7_exp['validation'] - bmw7_ctr_vl, range(-2, 3))

        if os.listdir(protobuf_dir):
            shutil.rmtree(protobuf_dir)
            os.mkdir(protobuf_dir)

    def test_run(self):
        protobuf_dir = dirs.get_protobuf_dir(BOT_ID)
        training_data_dir = dirs.get_training_data_dir(BOT_ID)
        if not os.listdir(training_data_dir):
            print("Cannot start test. No data in %s" & training_data_dir)
            return
        if not os.path.exists(protobuf_dir):
            os.mkdir(protobuf_dir)
        if os.listdir(protobuf_dir):
            shutil.rmtree(protobuf_dir)
            os.mkdir(protobuf_dir)

        converter.run(training_data_dir, protobuf_dir, 0.1)

        # Check if the labels.txt has been created
        self.assertTrue(os.path.isfile(os.path.join(protobuf_dir, 'labels.txt')))

        # Make sure the labels file contains as mainy files as the training data folder has subfolders
        with open(os.path.join(protobuf_dir, 'labels.txt')) as f:
            for lndx, dir in enumerate(os.listdir(training_data_dir)):
                pass
            for fndx, ln in enumerate(f):
                pass
            self.assertEqual(lndx, fndx)

        # Make sure there are 10
        protofiles = 0
        training_files = 0
        validation_files = 0
        for file in os.listdir(protobuf_dir):
            if file.endswith('.tfrecord'):
                protofiles += 1
            if 'train' in file:
                training_files += 1
            if 'validation' in file:
                validation_files += 1
        self.assertEqual(10, protofiles)
        self.assertEqual(5, training_files)
        self.assertEqual(5, validation_files)

        if os.listdir(protobuf_dir):
            shutil.rmtree(protobuf_dir)
            os.mkdir(protobuf_dir)


if __name__ == '__main__':
    unittest.main()
