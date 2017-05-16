import os
import tensorflow as tf

from cnn_server.server import file_service as dirs
from slim.datasets import dataset_utils
from slim.nets import nets_factory as network_factory
from slim.preprocessing import preprocessing_factory as preprocessing_factory

slim = tf.contrib.slim


def map_predictions_to_labels(bot_id, predictions, return_labels):
    """
    Utility function to map the output of the prediction endpoint to the corresponding labels
    :param bot_id: 
    :param predictions: 
    :param return_labels: 
    :return: 
    """
    labels = []
    for line in open(os.path.join(dirs.get_protobuf_dir(bot_id), 'labels.txt')):
        labels.append(line.split(':')[1].replace('\n', ''))

    # Get the indices of the n predictions with highest score
    top_n = predictions.argsort()[-return_labels:][::-1]

    lbls = [labels[ndx] for ndx in top_n]
    probabilities = predictions[top_n].tolist()
    return lbls, probabilities


def inference_on_image(bot_id, image_file, network_name='inception_v3', return_labels=1):
    """
    Loads the corresponding model checkpoint, network function and preprocessing routine based on bot_id and network_name,
    restores the graph and runs it to the prediction enpoint with the image as input
    :param bot_id: bot_id, used to reference to correct model directory
    :param image_file: reference to the temporary image file to be classified
    :param network_name: name of the network type to be used
    :param return_labels: number of labels to return
    :return: the top n labels with probabilities, where n = return_labels
    """

    # Get the model path
    model_path = dirs.get_model_data_dir(bot_id)

    # Get number of classes to predict
    protobuf_dir = dirs.get_protobuf_dir(bot_id)
    number_of_classes = dataset_utils.get_number_of_classes_by_labels(protobuf_dir)

    # Get the preprocessing and network construction functions
    preprocessing_fn = preprocessing_factory.get_preprocessing(network_name, is_training=False)
    network_fn = network_factory.get_network_fn(network_name, number_of_classes)

    # Process the temporary image file into a Tensor of shape [widht, height, channels]
    image_tensor = tf.gfile.FastGFile(image_file, 'rb').read()
    image_tensor = tf.image.decode_image(image_tensor, channels=0)

    # Perform preprocessing and reshape into [network.default_width, network.default_height, channels]
    network_default_size = network_fn.default_image_size
    image_tensor = preprocessing_fn(image_tensor, network_default_size, network_default_size)

    # Create an input batch of size one from the preprocessed image
    input_batch = tf.reshape(image_tensor, [1, 299, 299, 3])

    # Create the network up to the Predictions Endpoint
    logits, endpoints = network_fn(input_batch)

    # Create a Saver() object to restore the network from the last checkpoint
    restorer = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Restore the variables of the network from the last checkpoint and run the graph
        restorer.restore(sess, tf.train.latest_checkpoint(model_path))
        sess.run(endpoints)

        # Get the numpy array of predictions out of the
        predictions = endpoints['Predictions'].eval()[0]

    return map_predictions_to_labels(bot_id, predictions, return_labels)
