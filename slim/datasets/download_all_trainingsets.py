import os

import slim.datasets.download_training_data as dl
from cnn_server.server import file_service as dirs

## Script to download training data for transfer learning from csv files with Flickr Image URL

def _check_file(file):
    if not os.path.isfile(file):
        print("No such file %s" % file)
        return False
    return True


def download_all():
    if not _check_file(dirs.get_root_model_training_file()): return None
    if not _check_file(dirs.get_transfer_learning_file('car_types')): return None
    if not _check_file(dirs.get_transfer_learning_file('cars')): return None
    if not _check_file(dirs.get_transfer_learning_file('seasons')): return None
    # if not _check_file(dirs.get_transfer_learning_file('bmw_models')): return None

    dl.download_training_data('root', dirs.get_root_model_training_file(), resume_from=None)
    dl.download_training_data('car_types', dirs.get_transfer_learning_file('car_types'), resume_from=None)
    dl.download_training_data('cars', dirs.get_transfer_learning_file('cars'), resume_from=None)
    dl.download_training_data('seasons', dirs.get_transfer_learning_file('seasons'), resume_from=None)
    # dl.download_training_data('bmw_models', dirs.get_transfer_learning_file('bmw_models'), resume_from=None)



if __name__ == '__main__':
    download_all()
