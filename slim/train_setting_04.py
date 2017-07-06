from slim import train_setting_from_cars as train_from_cars
from slim import train_setting_from_imagenet as train_from_imagenet
from slim import train_setting_from_scratch as train_from_scratch

bots_setting_03 = ['cars', 'bmw_models', 'car_types', 'seasons']


# train_from_cars.train(4, 'bmw_models', hours=1)
# train_from_imagenet.train(4, 'bmw_models', hours=1)
# train_from_scratch.train(4, 'bmw_models', hours=1)

train_from_cars.train(4, 'cars', hours=1)
train_from_imagenet.train(4, 'cars', hours=1)
train_from_scratch.train(4, 'cars', hours=1)

train_from_cars.train(4, 'car_types', hours=1)
train_from_imagenet.train(4, 'car_types', hours=1)
train_from_scratch.train(4, 'car_types', hours=1)

train_from_cars.train(4, 'seasons', hours=1)
train_from_imagenet.train(4, 'seasons', hours=1)
train_from_scratch.train(4, 'seasons', hours=1)