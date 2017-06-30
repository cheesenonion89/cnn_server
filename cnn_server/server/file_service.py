import os

PROJECT_ROOT_DIR = '/home/markus/projects/cnn_server/'
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'training_data')
TRANSFER_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'transfer_learning')
PROTOBUF_DIR = os.path.join(PROJECT_ROOT_DIR, 'protobuf')
MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, 'model')
PERFORMANCE_DIR = os.path.join(PROJECT_ROOT_DIR, 'performance')
DATASET_DIR = os.path.join(PROJECT_ROOT_DIR, 'datasets')
DATASET_TRAIN_DIR = os.path.join(DATASET_DIR, 'training')
DATASET_TEST_DIR = os.path.join(DATASET_DIR, 'test')
DATASET_TRANSFER_DIR = os.path.join(DATASET_DIR, 'transfer learning')

FOLDER_PREFIX = 'bot'


def folder_name(bot_id):
    return '%s_%s/' % (FOLDER_PREFIX, bot_id)


def _create_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_training_data_dir(bot_id):
    return _create_if_not_exists(os.path.join(TRAINING_DATA_DIR, folder_name(bot_id)))


def get_transfer_setting_dir(path, transfer_setting):
    return _create_if_not_exists(os.path.join(path, 'setting_0%s' % transfer_setting))


def get_transfer_data_dir(bot_id, transfer_setting):
    return _create_if_not_exists(
        os.path.join(get_transfer_setting_dir(TRANSFER_DATA_DIR, transfer_setting), folder_name(bot_id)))


def get_transfer_proto_dir(bot_id, transfer_setting):
    return _create_if_not_exists(os.path.join(get_transfer_setting_dir(PROTOBUF_DIR, transfer_setting),
                                              folder_name(bot_id)))


def get_transfer_model_dir(bot_id, transfer_setting, suffix=''):
    return _create_if_not_exists(os.path.join(get_transfer_setting_dir(MODEL_DIR, transfer_setting),
                                              '%s%s' % (folder_name(bot_id).replace('/', ''), suffix)))


def get_readme_file(transfer_setting):
    return os.path.join(get_transfer_setting_dir(transfer_setting), 'README')


def get_protobuf_dir(bot_id):
    return _create_if_not_exists(os.path.join(PROTOBUF_DIR, folder_name(bot_id)))


def get_model_data_dir(bot_id):
    return os.path.join(MODEL_DIR, folder_name(bot_id))


def get_performance_data_dir(bot_id):
    return _create_if_not_exists(os.path.join(PERFORMANCE_DIR, folder_name(bot_id)))


def get_root_model_dir():
    return _create_if_not_exists(os.path.join(MODEL_DIR, folder_name('root')))


def get_imagenet_model_dir():
    return _create_if_not_exists(os.path.join(MODEL_DIR, folder_name('imagenet')))


def get_root_model_ckpt_path(model_name):
    root_model_ckpts = os.path.join(PROJECT_ROOT_DIR, 'root_model_checkpoints')
    root_model_ckpt_path = os.path.join(root_model_ckpts, model_name)
    if not os.path.exists(root_model_ckpt_path):
        print('There is no checkpoint for model %s' % model_name)
        return None
    else:
        return _create_if_not_exists(root_model_ckpt_path)


def get_test_root_model_dir():
    return _create_if_not_exists(os.path.join(MODEL_DIR, 'root_test'))


def get_dataset_dir():
    return _create_if_not_exists(DATASET_DIR)


def get_dataset_train_dir():
    return _create_if_not_exists(DATASET_TRAIN_DIR)


def get_dataset_transfer_dir():
    return _create_if_not_exists(DATASET_TRANSFER_DIR)


def get_root_model_training_file():
    return os.path.join(DATASET_TRAIN_DIR, 'initial_training_dataset_cars_conf70.csv')


def get_transfer_learning_file(dataset_name):
    transfer_datasets = {
        'bmw_models': 'transfer_dataset_bmw_models_conf70.csv',
        'car_types': 'transfer_dataset_car_types_conf70_sample.csv',
        'cars': 'transfer_dataset_cars_conf70.csv',
        'seasons': 'transfer_dataset_seasons_conf70_sample.csv'
    }
    dataset_file = transfer_datasets[dataset_name]
    if not dataset_file:
        print('no dataset file with name %s' % dataset_name)
        return None
    dataset_file_path = os.path.join(DATASET_TRANSFER_DIR, dataset_file)
    if not os.path.isfile(dataset_file_path):
        print('the dataset file %s does not exist' % dataset_file_path)
        return None
    return dataset_file_path


def get_transfer_learning_sample_file(sample_name):
    transfer_datasets = {
        'car_types': 'transfer_dataset_car_types_conf70_sample.csv',
        'seasons': 'transfer_dataset_seasons_conf70_sample.csv'
    }
    dataset_file = transfer_datasets[sample_name]
    dataset_file_path = os.path.join(DATASET_TRANSFER_DIR, dataset_file)
    return dataset_file_path
    # return _create_if_not_exists(dataset_file_path)


def get_test_training_file():
    return os.path.join(DATASET_TEST_DIR, 'test_dataset_cars_train.csv')


def get_bot_id_from_dir(dir_name: str):
    return dir_name.split('/')[-2].replace('bot_', '')
