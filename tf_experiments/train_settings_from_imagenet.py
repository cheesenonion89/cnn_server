from tf_experiments import train_setting_from_imagenet as trainer

bots_setting_02 = ['cars', 'bmw_models', 'car_types', 'seasons']

for bot_id in bots_setting_02:
    trainer.train(2, bot_id)