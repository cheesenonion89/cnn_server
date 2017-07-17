import os

import slim.datasets.convert_to_protobuf as converter
from cnn_server.server import file_service as dirs
import time

BOT_IDS = [
    'root',
    'bmw_models',
    'cars',
    'seasons',
    'car_types'
]


def _check_training_dir(tr_dir):
    if not os.path.isdir(tr_dir):
        print("No such directory %s" % tr_dir)
        return False
    if not os.listdir(tr_dir):
        print("Training dir %s is empty. Skipping." % tr_dir)
        return False
    return True


def _check_proto_dir(pr_dir):
    if not os.path.isdir(pr_dir):
        print("No such directory %s" % pr_dir)
        return False
    if os.listdir(pr_dir):
        print("Dataset is already present in %s" % pr_dir)
        return False
    return True


def _convert(bot_id):
    training_data_dir = dirs.get_training_data_dir(bot_id)
    protobuf_dir = dirs.get_protobuf_dir(bot_id)
    if _check_training_dir(training_data_dir) and _check_proto_dir(protobuf_dir):
        converter.run(training_data_dir, protobuf_dir)


def convert_all_trainingsets():
    for bot_id in BOT_IDS:
        print('Converting training data for %s' %bot_id)
        start_time = time.time()
        _convert(bot_id)
        print('Converted training data for %s in %s sec' %(bot_id, (time.time() - start_time)))

if __name__ == '__main__':
    convert_all_trainingsets()
