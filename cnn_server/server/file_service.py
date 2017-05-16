import os

PROJECT_ROOT_DIR = '/home/markus/projects/cnn_server/'
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'training_data')
PROTOBUF_DIR = os.path.join(PROJECT_ROOT_DIR, 'protobuf')
MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, 'model')
DATASET_DIR = os.path.join(PROJECT_ROOT_DIR, 'datasets')
DATASET_TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
DATASET_TEST_DIR = os.path.join(DATASET_DIR, 'test')
DATASET_TRANSFER_DIR = os.path.join(DATASET_DIR, 'transfer learning')

root_model_map = {
    'alex_net': 'alex_net',
    'cifar_net': 'cifar_net',
    'vgg_19': 'vgg_19',
    'vgg_19_pretrained': 'vgg_19_pretrained',
    'inception_v4': 'inception_v4',
    'inception_v4_pretrained': 'inception_v4_pretrained',
    'inception_resnet': 'inception_resnet',
    'inception_resnet_pretrained': 'inception_resnet_pretrained',
    'lenet': 'lenet',
    'resnet_v2_152': 'resnet_v2_152',
    'resnet_v2_152_pretrained': 'resnet_v2_152_pretrained'
}

FOLDER_PREFIX = 'bot'


def folder_name(bot_id):
    return '%s_%s/' % (FOLDER_PREFIX, bot_id)


def _create_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_training_data_dir(bot_id):
    return _create_if_not_exists(os.path.join(TRAINING_DATA_DIR, folder_name(bot_id)))


def get_protobuf_dir(bot_id):
    return _create_if_not_exists(os.path.join(PROTOBUF_DIR, folder_name(bot_id)))


def get_model_data_dir(bot_id):
    return _create_if_not_exists(os.path.join(MODEL_DIR, folder_name(bot_id)))


def get_root_model_dir(model_name=None):
    root_model_dir = os.path.join(MODEL_DIR, 'root')
    if not os.path.exists(root_model_dir):
        os.makedirs(root_model_dir)
    if not model_name:
        return _create_if_not_exists(os.path.join(root_model_dir, 'inception_v3'))
    else:
        if not model_name in root_model_map:
            print('model_name %s is not defined as root model' % model_name)
            return None
        root_model_dir = os.path.join(root_model_dir, model_name)
        if not os.path.exists(root_model_dir):
            print("Creating root model directory for model %s" % model_name)
            os.makedirs(root_model_dir)
        return _create_if_not_exists(root_model_dir)


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
    return _create_if_not_exists(os.path.join(DATASET_TRAIN_DIR, 'initial_training_dataset_cars_conf70.csv'))


def get_transfer_learning_file(dataset_name):
    transfer_datasets = {
        'bmw_models': 'transfer_dataset_bmw_models_conf70.csv',
        'car_types': 'transfer_dataset_car_types_conf70.csv',
        'cars': 'transfer_dataset_cars_conf70.csv',
        'seasons': 'transfer_dataset_seasons_conf70.csv'
    }
    dataset_file = transfer_datasets[dataset_name]
    if not dataset_file:
        print('no dataset file with name %s' % dataset_name)
        return None
    dataset_file_path = os.path.join(DATASET_TRANSFER_DIR, dataset_file)
    if not os.path.isfile(dataset_file_path):
        print('the dataset file %s does not exist' % dataset_file_path)
        return None
    return _create_if_not_exists(dataset_file_path)


def get_test_training_file():
    return os.path.join(DATASET_TEST_DIR, 'test_dataset_cars_train.csv')
