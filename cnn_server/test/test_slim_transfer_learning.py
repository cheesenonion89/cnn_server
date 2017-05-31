import json
import unittest
from unittest import TestCase

import base64
import os
import tempfile

import cnn_server.classification.classification_receive_handler as handler
import slim.transfer_image_classifier as transfer_learning
from cnn_server.server import file_service as dirs

TEST_BOT_ID = 'test'
FILES_DIR = 'files'


class TestTransferLearningImageClassifier(TestCase):
    def test_transfer_learning(self):
        # Root model to initialize from
        root_model_dir = dirs.get_test_root_model_dir()
        if not os.listdir(root_model_dir):
            print('root_model_dir %s empty. Cannot start test' % root_model_dir)
            return None
        if not os.path.isfile(os.path.join(root_model_dir, 'checkpoint')):
            print('No Checkpoint File in %s. Cannot start test.' % root_model_dir)
            return None

        # Folder to load the additional training data from
        bot_protobuf_dir = dirs.get_protobuf_dir(TEST_BOT_ID)
        if not os.path.isdir(bot_protobuf_dir):
            print('bot_protobuf_dir %s does not exist. Cannot start test' % bot_protobuf_dir)
            return None
        if not os.listdir(bot_protobuf_dir):
            print("bot_protobuf_dir %s is empty. Cannot start test." % bot_protobuf_dir)

        # Bot model folder to write the transfer learned model back to
        bot_model_dir = dirs.get_model_data_dir(TEST_BOT_ID)
        if not os.path.isdir(bot_model_dir):
            print('bot_model_dir %s does not exist. Cannot start test' % bot_model_dir)
            return None
        if os.listdir(bot_model_dir):
            print('bot_model_dir %s is not emtpy. Cannot start test.' % bot_model_dir)
            return None

        # Just run one step to make sure checkpoint files are written appropriately
        transfer_learning.transfer_learning(
            root_model_dir=root_model_dir,
            bot_model_dir=bot_model_dir,
            protobuf_dir=bot_protobuf_dir,
            max_train_time_sec=100,
            log_every_n_steps=2
        )

        # Check if the root model dir is still intact
        self.assertTrue(os.listdir(root_model_dir),
                        'root_model_dir %s is empty after transfer learning.' % root_model_dir)
        self.assertTrue(os.path.isfile(os.path.join(root_model_dir, 'checkpoint')),
                        'checkpoints file in root_model_dir %s is gone after transfer learning.' % root_model_dir)

        # Check if the bot model dir contains a model now
        self.assertTrue(os.listdir(bot_model_dir), 'bot_model_dir %s is empty after transfer learning' % bot_model_dir)
        self.assertTrue(os.path.isfile(os.path.join(bot_model_dir, 'checkpoint')),
                        'not checkpoints file in bot_model_dir %s after transfer learning' % bot_model_dir)

        # Mock a file for classification
        temp_file = tempfile.NamedTemporaryFile()
        temp_file.write(
            base64.b64encode(
                open(
                    os.path.join(FILES_DIR, 'tulip.jpg'), "rb"
                ).read()
            )
        )
        temp_file.seek(0)

        json_result, status = handler.handle_post(TEST_BOT_ID, temp_file.read(), return_labels=5)
        print(json_result)
        temp_file.close()

        self.assertTrue(json_result, 'Classification result is empty')

        json_result = json.loads(json_result)

        self.assertTrue(json_result['labels'], 'No labels in json result %s' % json_result)
        self.assertTrue(json_result['probabilities'], 'No predictions in json result %s' % json_result)

        print(json_result)

        # Clean the bot_model directory for next test run
        for file in os.listdir(bot_model_dir):
            file_path = os.path.join(bot_model_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)


"""
    def test_train(self):
        # Bot model folder to write the transfer learned model back to
        bot_model_dir = dirs.get_model_data_dir(TEST_BOT_ID)
        if not os.path.isdir(bot_model_dir):
            print('bot_model_dir %s does not exist. Cannot start test' % bot_model_dir)
            return None
        if os.listdir(bot_model_dir):
            print('bot_model_dir %s is not emtpy. Cannot start test.' % bot_model_dir)
            return None

        # Folder to load the additional training data from
        bot_protobuf_dir = dirs.get_protobuf_dir(TEST_BOT_ID)
        if not os.path.isdir(bot_protobuf_dir):
            print('bot_protobuf_dir %s does not exist. Cannot start test' % bot_protobuf_dir)
            return None
        if not os.listdir(bot_protobuf_dir):
            print("bot_protobuf_dir %s is empty. Cannot start test." % bot_protobuf_dir)

        transfer_learning.train(bot_model_dir=bot_model_dir, protobuf_dir=bot_protobuf_dir, max_train_time_sec=100)

        # Check if the bot model dir contains a model now
        self.assertTrue(os.listdir(bot_model_dir), 'bot_model_dir %s is empty after transfer learning' % bot_model_dir)
        self.assertTrue(os.path.isfile(os.path.join(bot_model_dir, 'checkpoint')),
                        'not checkpoints file in bot_model_dir %s after transfer learning' % bot_model_dir)
        for file in os.listdir(bot_model_dir):
            file_path = os.path.join(bot_model_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
"""

if __name__ == '__main__':
    unittest.main()
