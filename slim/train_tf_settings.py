from slim import train_tf_setting as trainer

bots_setting_01 = ['cars', 'bmw_models', 'car_types']
bots_setting_02 = ['cars', 'bmw_models', 'car_types', 'seasons']
settings = [bots_setting_01, bots_setting_02]


for index, bot_ids in enumerate(settings):
    setting_id = index + 1
    for bot_id in bot_ids:
        trainer.train(setting_id, bot_id)
