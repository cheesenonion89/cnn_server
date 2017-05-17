import os
import tensorflow as tf
import random

_FRACT_VALIDATION = 0.1
_NUM_SHARDS = 5
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


def run(training_data_dir, protobuf_dir, fract_validation=_FRACT_VALIDATION, num_shards=_NUM_SHARDS):
    """

        :param training_data_dir: 
        :param protobuf_dir: 
        :param num_validation: 
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

    random.seed(_RANDOM_SEED)
    random.shuffle(training_filenames)
    random.shuffle(validation_filenames)


