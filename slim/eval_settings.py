from slim import eval_image_classifier as evaluator

bot_ids = ['cars', 'bmw_models', 'car_types', 'seasons']
suffixes = ['', '_from_imagenet', '_from_scratch']

for bot_id in bot_ids:
    for suffix in suffixes:
        print("####################################################################################################\n")
        print("EVALUATING BOT: bot_%s%s" % (bot_id, suffix))
        print("\n####################################################################################################")
        evaluator.eval(bot_id=bot_id, bot_suffix=suffix, setting_id=8)
