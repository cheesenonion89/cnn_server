import sys
import time

import os
import pandas as pd
import re
import requests
import shutil

from cnn_server.server import file_service as dirs


def _create_label_folders(bot_id, labels, resume_from):
    training_data_dir = dirs.get_training_data_dir(bot_id)
    if os.listdir(training_data_dir) and not resume_from:
        print('Overwriting current training data in %s' % training_data_dir)
        shutil.rmtree(training_data_dir)
        os.mkdir(training_data_dir)
        for label in labels:
            os.mkdir(os.path.join(training_data_dir, label))
    if os.listdir(training_data_dir) and resume_from:
        print('Resuming from %s. Label Folders exist.' % resume_from)
    if not os.listdir(training_data_dir):
        print('Creating file structure for training data in %s' % training_data_dir)
        for label in labels:
            os.mkdir(os.path.join(training_data_dir, label))


def _get_image_path(bot_id, label, image_name, url):
    image_path = os.path.join(dirs.get_training_data_dir(bot_id), label)
    url_ending = re.sub(r"jpg.*", 'jpg', url.split('/')[-1])
    file_name = '%s_%s' % (image_name, url_ending)
    return os.path.join(image_path, file_name)


def _download_and_safe_image(url, image_path):
    na_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00'
    byte_content = requests.get(url).content
    if byte_content[0:10] == na_image_bytes:
        print('Image from %s is not available. Url is skipped.' % url)
        return False
    f = open(image_path, 'wb')
    f.write(byte_content)
    f.close()
    return True


def download_training_data(bot_id, from_csv, resume_from=None):
    """
    
    :param bot_id: 
    :param from_csv: 
    :param resume_from: 
    :return: 
    """
    training_data = pd.DataFrame().from_csv(from_csv, header=0, sep=';', index_col=None)
    if resume_from:
        print('Resuming training data iteration from %s' % resume_from)
        training_data = training_data.loc[int(resume_from):]

    print('labels %s' % pd.Series.unique(training_data.ix[:, 3]))
    _create_label_folders(bot_id, pd.Series.unique(training_data.ix[:, 3]), resume_from)

    thumbnail_ctr = 0
    image_ctr = 0
    na_ctr = 0
    start_time = time.time()
    for idx, row in training_data.iterrows():
        image_id = row['image_id']
        label = row[3]
        if not pd.isnull(row['thumbnail_url']):
            print('Downloading thumbnail %s from %s' % (idx, row['thumbnail_url']))
            # urllib.request.urlretrieve(row['thumbnail_url'],
            #                            _get_image_path(bot_id, label, image_id, row['thumbnail_url']))
            success = _download_and_safe_image(row['thumbnail_url'],
                                               _get_image_path(bot_id, label, image_id, row['thumbnail_url']))
            if not success:
                na_ctr += 1
            else:
                thumbnail_ctr += 1
        elif not pd.isnull(row['image_url']):
            print('Downloading image %s from %s' % (idx, row['image_url']))
            # urllib.request.urlretrieve(row['image_url'], _get_image_path(bot_id, label, image_id, row['image_url']))
            success = _download_and_safe_image(row['image_url'],
                                               _get_image_path(bot_id, label, image_id, row['image_url']))
            if not success:
                na_ctr += 1
            else:
                image_ctr += 1
        else:
            print('Skipping image %s with id %s. No download URL.' % (idx, image_id))
            pass
    print('Finished. Downloaded %s files in %s seconds.' % ((thumbnail_ctr + image_ctr), (time.time() - start_time)))
    print('Downloaded %s thumbnails and %s images. %s images were not available.' % (thumbnail_ctr, image_ctr, na_ctr))


"""
if __name__ == '__main__':
    args = [None, None, None]
    for idx, arg in enumerate(sys.argv):
        args[idx] = arg
    dataset_name = args[1]
    resume_from = args[2]

    if dataset_name:
        print('Downloading transfer dataset %s' % dataset_name)
        download_training_data(dataset_name, dirs.get_transfer_learning_file(dataset_name), resume_from)
    else:
        print('Downloading cars dataset for initial training')
        download_training_data('root', dirs.get_root_model_training_file(), resume_from)
"""