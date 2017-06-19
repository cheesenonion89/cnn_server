import time

import os
from shutil import copyfile

import slim.datasets.convert_to_protobuf as converter
from cnn_server.server import file_service as dirs

BOT_IDS = [
    'cars',
    'seasons',
    'car_types',
    'bmw_models'
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
        for file in os.listdir(pr_dir):
            path = os.path.join(pr_dir, file)
            try:
                if os.path.isfile(path):
                    os.unlink(path)
            except Exception as e:
                print(e)
        print("Cleared the protobuf folder %s" % pr_dir)
    return True


def _convert(bot_id, transfer_setting):
    training_data_dir = dirs.get_transfer_data_dir(bot_id, transfer_setting)
    protobuf_dir = dirs.get_transfer_proto_dir(bot_id, transfer_setting)
    '''
    readme_file = dirs.get_readme_file(transfer_setting)
    if not readme_file:
        print('No README found in %s' % training_data_dir)
    if readme_file:
        copyfile(readme_file, os.path.join(protobuf_dir, 'README'))
    '''
    if _check_training_dir(training_data_dir) and _check_proto_dir(protobuf_dir):
        converter.run(training_data_dir, protobuf_dir, fract_validation=0.2)


def convert_all_trainingsets(transfer_setting):
    for bot_id in BOT_IDS:
        print('Converting training data for %s' % bot_id)
        start_time = time.time()
        _convert(bot_id, transfer_setting)
        print('Converted training data for %s in %s sec' % (bot_id, (time.time() - start_time)))


if __name__ == '__main__':
    convert_all_trainingsets(2)
