import slim.transfer_learn_image_classifier as tl
from cnn_server.server import file_service as dirs

_TRAINING_DATA_ROOT_FOLDER = dirs.get_protobuf_dir('root')

networks_map = {
    'alex_net': 'alexnet_v2',
    'cifar_net': 'cifarnet',
    'vgg_19': 'vgg_19',
    'vgg_19_pretrained': 'vgg_19',
    'inception_v4': 'inception_v4',
    'inception_v4_pretrained': 'inception_v4',
    'inception_resnet': 'inception_resnet_v2',
    'inception_resnet_pretrained': 'inception_resnet_v2',
    'lenet': 'lenet',
    'resnet_v2_152': 'resnet_v2_152',
    'resnet_v2_152_pretrained': 'resnet_v2_152'
}


def train_root_model(root_model, max_train_time=None):
    root_model_path = dirs.get_root_model_dir(model_name=root_model)
    network_name = networks_map[root_model]
    root_model_ckpt_path = None
    if root_model.endswith('_pretrained'):
        root_model_ckpt_path = dirs.get_root_model_ckpt_path(network_name)
        if not root_model_ckpt_path:
            pass

    tl.train(
        bot_model_dir=root_model_path,
        protobuf_dir=_TRAINING_DATA_ROOT_FOLDER,
        root_model_dir=root_model_ckpt_path,
        model_name=network_name,
        max_train_time_sec=max_train_time
    )


def train_root_models(max_train_time_per_model=None):
    for root_model in dirs.root_model_map:
        train_root_model(root_model, max_train_time=max_train_time_per_model)
