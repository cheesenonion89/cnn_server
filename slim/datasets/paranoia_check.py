from cnn_server.server import file_service as dirs
import os

def _check_file(file):
    if not os.path.isfile(file):
        print("No such file %s" % file)
        return False
    return True


def check():
    if not _check_file(dirs.get_root_model_training_file()): return None
    if not _check_file(dirs.get_transfer_learning_file('car_types')): return None
    if not _check_file(dirs.get_transfer_learning_file('cars')): return None
    if not _check_file(dirs.get_transfer_learning_file('seasons')): return None
    print('All files are there')

check()
