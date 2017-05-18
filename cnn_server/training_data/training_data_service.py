import zipfile
from zipfile import ZipFile

import os

import cnn_server.server.file_service as dirs
from slim.datasets import convert_to_protobuf as converter


def validate_training_data(file):
    """
    
    :param path: 
    :return: 
    """
    if not zipfile.is_zipfile(file):
        return False

    with ZipFile(file) as zip:
        prev_label_dir = False
        members = len(zip.namelist())
        if not zip.namelist():
            return False
        for index, member in enumerate(zip.namelist()):

            # If the folder structure goes deeper than one subfolder, return False
            if len(member.split('/')) > 2:
                return False

            label = member.split('/')[0]
            file = None

            if len(member.split('/')) == 2:
                file = member.split('/')[1]

            # If the label is empty, return False
            if not label:
                return False

            # If the previous folder has been a directory and the current one too, we have an empty directory
            if prev_label_dir and not file:
                return False
            elif label and not file:
                prev_label_dir = True
            else:
                prev_label_dir = False

            # If the last member in the list is a directory, it must be empty
            if not file and index + 1 == members:
                return False

            # If a file does not end with any of the valid file endings, it is invalid
            if file and not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                return False

    return True


def create_training_data_dir(bot_id: int, training_data_file):
    """
    Verify that the files training data directory is there and empty and write the zipped training data to it.
    
    :param bot_id: 
    :param zip_file: 
    :return: 
    """
    bot_training_dir = dirs.get_training_data_dir(bot_id)

    # If the training data directory is not there, create it
    if not os.path.exists(bot_training_dir):
        os.mkdir(bot_training_dir)

    # If the training data directory already contains data, don't do anything
    if os.listdir(bot_training_dir):
        return False

    # Extract the contents of the zip file to the training data directory

    ZipFile(training_data_file).extractall(bot_training_dir)
    return True


def write_to_protobuffer(bot_id: int):
    """
    
    :param bot_id: 
    :param net: 
    :return: 
    """

    bot_training_data_dir = dirs.get_training_data_dir(bot_id)

    if not os.path.exists(bot_training_data_dir):
        return False

    bot_protobuf_dir = dirs.get_protobuf_dir(bot_id)

    if not os.path.exists(bot_protobuf_dir):
        os.mkdir(bot_protobuf_dir)

    converter.run(bot_training_data_dir, bot_protobuf_dir)

    return True




