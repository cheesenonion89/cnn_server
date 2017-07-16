import math
import sys

import os
import random
import tensorflow as tf

from slim.datasets import dataset_utils

_FRACT_VALIDATION = 0.2
_NUM_SHARDS = 1
_RANDOM_SEED = 0


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_dataset_filename(protobuf_dir, split_name, shard_id, num_shards):
    output_filename = 'sample_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, num_shards)
    return os.path.join(protobuf_dir, output_filename)


def _dataset_exists(dataset_dir, num_shards):
    for split_name in ['train', 'validation']:
        for shard_id in range(num_shards):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id, num_shards)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def _get_filenames_and_classes(training_data_dir, fract_validation):
    class_names = []
    training_files = []
    validation_files = []
    # Iterate over all label folders in the training data directory
    for filename in os.listdir(training_data_dir):
        class_files = []
        class_folder_path = os.path.join(training_data_dir, filename)
        # Collect the class names from the directory names
        if os.path.isdir(class_folder_path):
            class_names.append(filename)
        # Iterate over all files in a class directory and append them to the current list of class_files
        for file in os.listdir(class_folder_path):
            file_path = os.path.join(class_folder_path, file)
            class_files.append(file_path)

        # Make sure that from each class folder the train and validation ratio is present in the respective sets
        random.seed(_RANDOM_SEED)
        random.shuffle(class_files)
        num_validation = int(len(class_files) * fract_validation)
        training_files.extend(class_files[num_validation:])
        validation_files.extend(class_files[:num_validation])

    return training_files, validation_files, class_names


def _convert_dataset(split_name, filenames, class_names_to_ids, protobuf_dir, num_shards):
    """

    :param split_name: 
    :param filenames: 
    :param class_names_to_ids: 
    :param protobuf_dir: 
    :return: 
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(num_shards):
                output_filename = _get_dataset_filename(
                    protobuf_dir, split_name, shard_id, num_shards)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        print("Converting image %s" % filenames[i])
                        try:
                            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        except Exception as e:
                            print(e)
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def run(training_data_dir, protobuf_dir, fract_validation=_FRACT_VALIDATION, num_shards=_NUM_SHARDS):
    """
    
    :param training_data_dir: 
    :param protobuf_dir: 
    :param fract_validation: 
    :param num_shards: 
    :return: 
    """
    if not tf.gfile.Exists(protobuf_dir):
        tf.gfile.MakeDirs(protobuf_dir)

    if _dataset_exists(protobuf_dir, num_shards):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    training_filenames, validation_filenames, class_names = _get_filenames_and_classes(training_data_dir,
                                                                                       fract_validation)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    random.seed(_RANDOM_SEED)
    random.shuffle(training_filenames)
    random.shuffle(validation_filenames)

    _convert_dataset('train', training_filenames, class_names_to_ids, protobuf_dir, num_shards)
    _convert_dataset('validation', validation_filenames, class_names_to_ids, protobuf_dir, num_shards)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, protobuf_dir)

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the dataset!')
