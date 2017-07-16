from tf_experiments import train_setting_from_scratch as train_from_scratch, \
    train_setting_from_imagenet as train_from_imagenet, train_setting_from_cars as train_from_cars

bots_setting_03 = ['cars', 'bmw_models', 'car_types', 'seasons']


# train_from_cars.train(3, 'cars', hours=1)
# train_from_imagenet.train(3, 'cars', hours=1)
train_from_scratch.train(3, 'cars', hours=1)
# train_from_cars.train(3, 'bmw_models', hours=1)
train_from_imagenet.train(3, 'bmw_models', hours=1)
train_from_scratch.train(3, 'bmw_models', hours=1)
train_from_cars.train(3, 'car_types', hours=1)
train_from_imagenet.train(3, 'car_types', hours=1)
train_from_scratch.train(3, 'car_types', hours=1)
train_from_cars.train(3, 'seasons', hours=1)
train_from_imagenet.train(3, 'seasons', hours=1)
train_from_scratch.train(3, 'seasons', hours=1)
