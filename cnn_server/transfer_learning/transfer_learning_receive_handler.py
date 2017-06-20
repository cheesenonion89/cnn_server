from cnn_server.transfer_learning import transfer_learning_service as service

_DEFAULT_TRAIN_TIME = (60 * 60)


def handle_put(bot_id, test=False, max_train_time=_DEFAULT_TRAIN_TIME):
    if service.train(bot_id, test=test, max_train_time=max_train_time):
        return 'Transfer Learning of Bot %s was successful.' % bot_id, 200
    else:
        return 'Error in Transfer Learning of Bot %s' % bot_id, 500
