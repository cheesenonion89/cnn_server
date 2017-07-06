import os

from cnn_server.server import file_service as dirs
from slim import transfer_image_classifier as trainer


def _check_root_model_dir(root_model_dir):
    if not os.listdir(root_model_dir):
        print("No root model in %s" % root_model_dir)
        return False
    return True


def _check_protobuf_dir(protobuf_dir):
    if not os.listdir(protobuf_dir):
        print("No protobuffer in %s" % protobuf_dir)
        return False
    return True


def _check_bot_model_dir(bot_model_dir):
    if os.listdir(bot_model_dir):
        print('Bot Model Dir %s is not empty' % bot_model_dir)
        return False
    return True


def train(setting_id, bot_id, hours=24, minutes=60, seconds=60, summary_secs=60):
    root_model_dir = dirs.get_imagenet_model_dir()

    bot_protobuf_dir = dirs.get_transfer_proto_dir(bot_id, setting_id)
    bot_model_dir = dirs.get_transfer_model_dir(bot_id, setting_id, '_from_imagenet')

    _check_bot_model_dir(bot_model_dir)
    if _check_protobuf_dir(bot_protobuf_dir) and _check_root_model_dir(root_model_dir) and _check_bot_model_dir(
            bot_model_dir):
        print("""
            STARTING TRANSFER LEARNING
            ROOT MODEL:\t%s
            TARGET MODEL:\t%s
            DATASET:\t%s
        """ % (root_model_dir, bot_model_dir, bot_protobuf_dir))

        trainer.transfer_learning(
            root_model_dir=root_model_dir,
            bot_model_dir=bot_model_dir,
            protobuf_dir=bot_protobuf_dir,
            model_name='inception_v4',
            checkpoint_exclude_scopes=['InceptionV4/Logits', 'InceptionV4/AuxLogits'],
            trainable_scopes=['InceptionV4/Logits', 'InceptionV4/AuxLogits'],
            max_train_time_sec=(seconds * minutes * hours),  # seconds * minutes * hours * days
            optimization_params=None,
            log_every_n_steps=10,
            save_summaries_secs=summary_secs
        )
