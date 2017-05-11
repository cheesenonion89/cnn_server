import unittest
from unittest import TestCase
import os
from cnn_server.server import file_service as dirs


class TestServer(TestCase):
    def test_get_root_model_dir(self):
        root_model_dir = dirs.get_root_model_dir(model_name='inception_v4')
        self.assertTrue(os.path.exists(root_model_dir))

        self.assertFalse(dirs.get_root_model_dir(model_name='hokuspokus'))


if __name__ == '__main__':
    unittest.main()
