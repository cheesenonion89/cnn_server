from slim import train_setting_from_cars as trainer

bots_setting_01 = ['cars', 'bmw_models', 'car_types']
bots_setting_02 = ['cars', 'bmw_models', 'car_types', 'seasons']
settings = [bots_setting_02]

# for bot_id in bots_setting_01:
#    trainer.train(1, bot_id)

for bot_id in bots_setting_02:
    trainer.train(2, bot_id)
