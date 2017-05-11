import os

PROJECT_ROOT_DIR = '/home/markus/projects/cnn_server/'
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'training_data')
PROTOBUF_DIR = os.path.join(PROJECT_ROOT_DIR, 'protobuf')
MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, 'model')

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


def get_training_data_dir(bot_id):
    return os.path.join(TRAINING_DATA_DIR, folder_name(bot_id))


def get_protobuf_dir(bot_id):
    return os.path.join(PROTOBUF_DIR, folder_name(bot_id))


def get_model_data_dir(bot_id):
    return os.path.join(MODEL_DIR, folder_name(bot_id))


def get_root_model_dir(model_name=None):
    root_model_dir = os.path.join(MODEL_DIR, 'root')
    if not os.path.exists(root_model_dir):
        os.makedirs(root_model_dir)
    if not model_name:
        return root_model_dir
    else:
        if not model_name in root_model_map:
            print('model_name %s is not defined as root model' % model_name)
            return None
        root_model_dir = os.path.join(root_model_dir, model_name)
        if not os.path.exists(root_model_dir):
            print("Creating root model directory for model %s" % model_name)
            os.makedirs(root_model_dir)
        return root_model_dir


def get_test_root_model_dir():
    return os.path.join(MODEL_DIR, 'root_test')
