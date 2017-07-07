from slim import train_setting_from_cars as train_from_cars
from slim import train_setting_from_imagenet as train_from_imagenet
from slim import train_setting_from_scratch as train_from_scratch

# train_from_cars.train(9, 'cars', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(9, 'cars', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(9, 'cars', hours=1, minutes=10, summary_secs=20)
train_from_cars.train(9, 'bmw_models', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(9, 'bmw_models', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(9, 'bmw_models', hours=1, minutes=10, summary_secs=20)
train_from_cars.train(9, 'car_types', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(9, 'car_types', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(9, 'car_types', hours=1, minutes=10, summary_secs=20)
train_from_cars.train(9, 'seasons', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(9, 'seasons', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(9, 'seasons', hours=1, minutes=10, summary_secs=20)
