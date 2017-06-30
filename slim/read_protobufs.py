import math

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from cnn_server.server import file_service as dirs
from slim.datasets import dataset_factory
from slim.nets import nets_factory
from slim.preprocessing import preprocessing_factory

DATASET_NAME = 'bot'
MODEL_NAME = 'inception_v4'

LABELS_OFFSET = 0
BATCH_SIZE = 50
MAX_NUM_BATCHES = None
EVAL_IMAGE_SIZE = None
NUM_THREADS = 4


def _check_dir(dir_path):
    if not dir_path:
        raise ValueError('The directory string is empty')
    if not os.path.isdir(dir_path):
        raise ValueError('%s is not a directory' % dir_path)
    if not os.listdir(dir_path):
        raise ValueError('%s is empty' % dir_path)


def eval(bot_id, bot_suffix, setting_id=None, dataset_split='train', dataset_name='bot', model_name='inception_v4',
         preprocessing=None,
         moving_average_decay=None, tf_master=''):
    full_id = bot_id + bot_suffix
    if setting_id:
        protobuf_dir = dirs.get_transfer_proto_dir(bot_id, setting_id)
        model_dir = dirs.get_transfer_model_dir(full_id, setting_id)
    else:
        protobuf_dir = dirs.get_protobuf_dir(bot_id)
        model_dir = dirs.get_model_data_dir(full_id)

    _check_dir(protobuf_dir)
    _check_dir(model_dir)

    print("READIND FROM %s AND %s" %(protobuf_dir, model_dir))

    performance_data_dir = dirs.get_performance_data_dir(bot_id)
    #    if os.listdir(performance_data_dir):
    #        raise ValueError('%s is not empty' % performance_data_dir)

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            dataset_name, dataset_split, protobuf_dir)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            model_name,
            num_classes=(dataset.num_classes - LABELS_OFFSET),
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * BATCH_SIZE,
            common_queue_min=BATCH_SIZE)
        [image, label] = provider.get(['image', 'label'])
        label -= LABELS_OFFSET

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = preprocessing or model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = EVAL_IMAGE_SIZE or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=NUM_THREADS,
            capacity=5 * BATCH_SIZE)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(images)

        if moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # TODO(sguada) use num_epochs=1
        if MAX_NUM_BATCHES:
            num_batches = MAX_NUM_BATCHES
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(BATCH_SIZE))

        if tf.gfile.IsDirectory(model_dir):
            checkpoint_path = tf.train.latest_checkpoint(model_dir)
        else:
            checkpoint_path = model_dir

        print(dataset.num_samples)
        print(dataset.num_classes)

eval('bmw_models', '', 2)