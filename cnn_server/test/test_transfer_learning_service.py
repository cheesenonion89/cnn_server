import json
import unittest
from unittest import TestCase

import base64
import os
import tempfile

import cnn_server.classification.classification_receive_handler as handler
import cnn_server.transfer_learning.transfer_learning_service as service
from cnn_server.server import file_service as dirs

TEST_BOT_ID = 'test'
FILES_DIR = 'files'


class TestTransferLearningService(TestCase):
    def test_train(self):
        service.train(TEST_BOT_ID, test=True, max_train_time=100)

        bot_model_dir = dirs.get_model_data_dir(TEST_BOT_ID)

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


if __name__ == '__main__':
    unittest.main()
