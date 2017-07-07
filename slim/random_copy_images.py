import os
from random import randint
from shutil import copyfile

from cnn_server.server import file_service as dirs

_SAMPLE_SIZE = 1
_SETTING = 27

bot_ids = ['cars', 'bmw_models', 'car_types', 'seasons']

car_classes = ['lamborghini', 'mazda', 'peugeot']
bmw_classes = ['bmw3', 'bmw5', 'bmw7']
car_type_classes = ['armored car', 'formula one car', 'muscle car']
season_classes = ['winter', 'spring', 'autumn']


def sample(lst, n):
    sample = []
    ctr = 0
    N = len(lst)
    while ctr < n:
        index = randint(0, N-1)
        sample.append(lst.pop(index))
        N = len(lst)
        print('Lentgh of list: %s \n Length of sample: %s' % (N, len(sample)))
        ctr += 1
    return sample


for car_class in car_classes:
    training_dir = os.path.join(dirs.get_training_data_dir('cars'), car_class)
    transfer_dir = os.path.join(dirs.get_transfer_data_dir('cars', _SETTING), car_class)

    images = os.listdir(training_dir)
    image_sample = sample(images, _SAMPLE_SIZE)

    for image in image_sample:
        copyfile(os.path.join(training_dir, image), os.path.join(transfer_dir, image))

for bmw_class in bmw_classes:
    training_dir = os.path.join(dirs.get_training_data_dir('bmw_models'), bmw_class)
    transfer_dir = os.path.join(dirs.get_transfer_data_dir('bmw_models', _SETTING), bmw_class)

    images = os.listdir(training_dir)
    image_sample = sample(images, _SAMPLE_SIZE)

    for image in image_sample:
        copyfile(os.path.join(training_dir, image), os.path.join(transfer_dir, image))

for car_type_class in car_type_classes:
    training_dir = os.path.join(dirs.get_training_data_dir('car_types'), car_type_class)
    transfer_dir = os.path.join(dirs.get_transfer_data_dir('car_types', _SETTING), car_type_class)

    images = os.listdir(training_dir)
    image_sample = sample(images, _SAMPLE_SIZE)

    for image in image_sample:
        copyfile(os.path.join(training_dir, image), os.path.join(transfer_dir, image))

for season_class in season_classes:
    training_dir = os.path.join(dirs.get_training_data_dir('seasons'), season_class)
    transfer_dir = os.path.join(dirs.get_transfer_data_dir('seasons', _SETTING), season_class)

    images = os.listdir(training_dir)
    image_sample = sample(images, _SAMPLE_SIZE)

    for image in image_sample:
        copyfile(os.path.join(training_dir, image), os.path.join(transfer_dir, image))
