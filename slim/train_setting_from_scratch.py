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
    '''
    print("Cleaning Bot Model Dir %s" % bot_model_dir)
    for file in os.listdir(bot_model_dir):
        path = os.path.join(bot_model_dir, file)
        try:
            if os.path.isfile(path):
                os.unlink(path)
        except Exception as e:
            print(e)
    '''
    return True


def train(setting_id, bot_id):
    # root_model_dir = dirs.get_root_model_dir()

    bot_protobuf_dir = dirs.get_transfer_proto_dir(bot_id, setting_id)
    bot_model_dir = dirs.get_transfer_model_dir(bot_id, setting_id, '_from_scratch')

    print(bot_model_dir)

    if _check_protobuf_dir(bot_protobuf_dir) and _check_bot_model_dir(bot_model_dir):
        print("""
            STARTING TRANSFER LEARNING
            ROOT MODEL:\t%s
            TARGET MODEL:\t%s
            DATASET:\t%s
        """ % ('None', bot_model_dir, bot_protobuf_dir))

        trainer.train(
            bot_model_dir=bot_model_dir,
            protobuf_dir=bot_protobuf_dir,
            model_name='inception_v4',
            max_train_time_sec=(60 * 60 * 24),  # seconds * minutes * hours * days
            optimization_params=None,
            log_every_n_steps=10,
            save_summaries_secs=600
        )
