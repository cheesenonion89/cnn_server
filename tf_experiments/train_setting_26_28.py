from tf_experiments import train_setting_from_scratch as train_from_scratch, \
    train_setting_from_imagenet as train_from_imagenet, train_setting_from_cars as train_from_cars

train_from_cars.train(26, 'cars', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(26, 'cars', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(26, 'cars', hours=1, minutes=10, summary_secs=20)
train_from_cars.train(26, 'bmw_models', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(26, 'bmw_models', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(26, 'bmw_models', hours=1, minutes=10, summary_secs=20)
train_from_cars.train(26, 'car_types', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(26, 'car_types', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(26, 'car_types', hours=1, minutes=10, summary_secs=20)
train_from_cars.train(26, 'seasons', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(26, 'seasons', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(26, 'seasons', hours=1, minutes=10, summary_secs=20)


train_from_cars.train(28, 'cars', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(28, 'cars', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(28, 'cars', hours=1, minutes=10, summary_secs=20)
train_from_cars.train(28, 'bmw_models', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(28, 'bmw_models', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(28, 'bmw_models', hours=1, minutes=10, summary_secs=20)
train_from_cars.train(28, 'car_types', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(28, 'car_types', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(28, 'car_types', hours=1, minutes=10, summary_secs=20)
train_from_cars.train(28, 'seasons', hours=1, minutes=10, summary_secs=20)
train_from_imagenet.train(28, 'seasons', hours=1, minutes=10, summary_secs=20)
train_from_scratch.train(28, 'seasons', hours=1, minutes=10, summary_secs=20)