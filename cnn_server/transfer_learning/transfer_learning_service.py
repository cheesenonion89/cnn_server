import os

import slim.transfer_image_classifier as transfer_learning
from cnn_server.server import file_service as dirs

'''
Interacts with the slim package and the file service to start the transfer learning with the correct root model, the 
correct training data and write the resulting model back to the correct folder in \model
'''

def train(bot_id, test=False, max_train_time=None):
    if test:
        max_train_time = 60 # If we run a test, train for one minute only

    root_model_dir = dirs.get_root_model_dir()
    bot_model_dir = dirs.get_model_data_dir(bot_id)
    bot_protobuf_dir = dirs.get_protobuf_dir(bot_id)

    # root_model_dir must exist, not be empty and contain a checkpoints file
    if not os.path.exists(root_model_dir):
        print('root_model_dir %s does not exist' % root_model_dir)
        return False
    if not os.listdir(root_model_dir):
        print('root_model_dir %s is empty' % root_model_dir)
        return False
    if not os.path.isfile(os.path.join(root_model_dir, 'checkpoint')):
        print('no checkpoint files in root_model_dir %s' % root_model_dir)
        return False

    # bot_model_dir must exist and be empty
    if not os.path.exists(bot_model_dir):
        print('bot_model_dir %s does not exist' % bot_model_dir)
        return False
    if os.listdir(bot_model_dir):
        print('bot_model_dir %s is not empty' % bot_model_dir)
        return False

    # bot_protobuf_dir must exist and not be empty
    if not os.path.exists(bot_protobuf_dir):
        print('bot_protobuf_dir %s does not exist' % bot_protobuf_dir)
        return False
    if not os.listdir(bot_protobuf_dir):
        print('bot_protobuf_dir %s does not contain training data' % bot_protobuf_dir)
        return False

    transfer_learning.transfer_learning(
        root_model_dir=root_model_dir,
        bot_model_dir=bot_model_dir,
        protobuf_dir=bot_protobuf_dir,
        dataset_name='bot',
        dataset_split_name='train',
        model_name='inception_v4',
        checkpoint_exclude_scopes=['InceptionV4/Logits', 'InceptionV4/AuxLogits'],
        trainable_scopes=['InceptionV4/Logits', 'InceptionV4/AuxLogits'],
        max_train_time_sec=max_train_time
    )

    # After Transfer Learning bot_model_dir must exist, not be empty and contain a checkpoint file
    if not os.path.exists(bot_model_dir):
        print('bot_model_dir %s does not exist after transfer learning' % bot_model_dir)
        return False
    if not os.listdir(bot_model_dir):
        print('bot_model_dir %s is empty after transfer learning' % bot_model_dir)
        return False
    if not os.path.isfile(os.path.join(bot_model_dir, 'checkpoint')):
        print('no checkpoint file in bot_model_dir %s after transfer learning' % bot_model_dir)

    # TODO: Implement proper validation of the createed model file: read ckpt path from first line and lookup in folder
    return True
