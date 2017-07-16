import time

import os
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging

from slim.datasets import dataset_factory
from slim.deployment import model_deploy
from slim.nets import nets_factory
from slim.preprocessing import preprocessing_factory

slim = tf.contrib.slim

### PERFORMANCE PARAMETERS
# Model Deployment
_NUM_CLONES = 1
_CLONE_ON_CPU = False
_TASK = 0
_WORKER_REPLICAS = 1
_NUM_PS_TASKS = 0

# Preprocessing
_NUM_READERS = 4
_BATCH_SIZE = 1
_NUM_PREPROCESSING_THREADS = 4

# Loss Function
_LABEL_SMOOTHING = 0.0

# Training
_SYNC_REPLICAS = False
_REPLICAS_TO_AGGREGATE = 1
_MASTER = ''
_MAX_NUMBER_OF_STEPS = None
_MAX_TRAIN_TIME_SECONDS = 86400  # By default training is run for max. 24 hours
_LOG_EVERY_N_STEPS = 10
_SAVE_SUMMARRIES_SECS = 600
_SAVE_INTERNAL_SECS = 600

# OPTIMIZER
_OPTIMIZER = 'rmsprop'
_OPT_EPSILON = 1.0

_ADADELTA_RHO = 0.95

_ADAGRAD_INITIAL_ACCUMULATOR_VALUE = 0.1

_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999

_RMSPROP_DECAY = 0.9

_FTRL_LEARNING_RATE_POWER = 0.01
_POWER = -0.5
_FTRL_INITIAL_ACCUMULATOR_VALUE = 0.1
_FTRL_L1 = 0.0
_FTRL_L2 = 0.0

# NETWORK FUNCTION
_LABELS_OFFSET = 0

# TRANSFER_LEARNING
_TRAINABLE_SCOPES = None
_CHECKPOINT_EXCLUDE_SCOPES = None
_IGNORE_MISSING_VARS = False

# ERROR FUNCTION / OPTIMIIZATION PARAMETERS

_MOVING_AVERAGE_DECAY = None  # Smoothing fator
_NUM_EPOCHS_PER_DECAY = 2.0

_LEARNING_RATE = 0.01
_LEARNING_RATE_DECAY_FACTOR = 0.94
_END_LEARNING_RATE = 0.0001
_LEARNING_RATE_DECAY_TYPE = 'exponential'

_MOMENTUM = 0.9  # 'Soften' the steepness of the gradient
_WEIGHT_DECAY = 0.00004  # Regularization of the weights, penalizes large weights, avoid overfitting

_DROPOUT_KEEP_PROB = None

OPTIMIZATION_PARAMS = {
    'moving_average_decay': _MOVING_AVERAGE_DECAY,
    'num_epochs_per_decay': _NUM_EPOCHS_PER_DECAY,

    'learning_rate': _LEARNING_RATE,
    'learning_rate_decay_factor': _LEARNING_RATE_DECAY_FACTOR,
    'end_learning_rate': _END_LEARNING_RATE,
    'learning_rate_decay_type': _LEARNING_RATE_DECAY_TYPE,

    'momentum': _MOMENTUM,
    'weight_decay': _WEIGHT_DECAY,

    'dropout_keep_prob': _DROPOUT_KEEP_PROB
}


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """
    
    :param num_samples_per_epoch: 
    :param global_step: 
    :return: 
    """
    decay_steps = int(num_samples_per_epoch / _BATCH_SIZE *
                      OPTIMIZATION_PARAMS['num_epochs_per_decay'])
    if _SYNC_REPLICAS:
        decay_steps /= _REPLICAS_TO_AGGREGATE

    if OPTIMIZATION_PARAMS['learning_rate_decay_type'] == 'exponential':
        return tf.train.exponential_decay(OPTIMIZATION_PARAMS['learning_rate'],
                                          global_step,
                                          decay_steps,
                                          OPTIMIZATION_PARAMS['learning_rate_decay_factor'],
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif OPTIMIZATION_PARAMS['learning_rate_decay_type'] == 'fixed':
        return tf.constant(OPTIMIZATION_PARAMS['learning_rate'], name='fixed_learning_rate')
    elif OPTIMIZATION_PARAMS['learning_rate_decay_type'] == 'polynomial':
        return tf.train.polynomial_decay(OPTIMIZATION_PARAMS['learning_rate'],
                                         global_step,
                                         decay_steps,
                                         OPTIMIZATION_PARAMS['end_learning_rate'],
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         OPTIMIZATION_PARAMS['learning_rate_decay_type'])


def _configure_optimizer(learning_rate):
    """
    
    :param learning_rate: 
    :return: 
    """
    if _OPTIMIZER == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=_ADADELTA_RHO,
            epsilon=_OPT_EPSILON)
    elif _OPTIMIZER == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=_ADAGRAD_INITIAL_ACCUMULATOR_VALUE)
    elif _OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=_ADAM_BETA1,
            beta2=_ADAM_BETA2,
            epsilon=_OPT_EPSILON)
    elif _OPTIMIZER == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=_FTRL_LEARNING_RATE_POWER,
            initial_accumulator_value=_FTRL_INITIAL_ACCUMULATOR_VALUE,
            l1_regularization_strength=_FTRL_L1,
            l2_regularization_strength=_FTRL_L2)
    elif _OPTIMIZER == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=OPTIMIZATION_PARAMS['momentum'],
            name='Momentum')
    elif _OPTIMIZER == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=_RMSPROP_DECAY,
            momentum=OPTIMIZATION_PARAMS['momentum'],
            epsilon=_OPT_EPSILON)
    elif _OPTIMIZER == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', _OPTIMIZER)
    return optimizer


def _get_init_fn(root_model_dir, bot_model_dir, checkpoint_exclude_scopes):
    """
    
    :param model_dir: 
    :param protobuf_dir: 
    :return: 
    """
    if root_model_dir is None:
        return None

    # Warn the user if a checkpoint exists in the bot_model_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(bot_model_dir):
        tf.logging.info(
            'Ignoring root_model_dir because a checkpoint already exists in %s'
            % bot_model_dir)
        return None

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(root_model_dir):
        checkpoint_path = tf.train.latest_checkpoint(root_model_dir)
    else:
        checkpoint_path = root_model_dir

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=_IGNORE_MISSING_VARS)


def _get_variables_to_train(trainable_scopes):
    """
    
    :return: 
    """
    if trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


# Remove number of steps from .learn call and define own kwargs
def _train_step_kwargs(logdir, max_train_time_seconds=_MAX_TRAIN_TIME_SECONDS, should_log=True,
                       log_every_n_steps=_LOG_EVERY_N_STEPS,
                       should_trace=False):
    train_step_kwargs = {}
    if logdir:
        train_step_kwargs['logdir'] = logdir

    if should_log:
        train_step_kwargs['should_log'] = should_log
    if should_trace:
        train_step_kwargs['should_trace'] = should_trace

    if log_every_n_steps:
        train_step_kwargs['log_every_n_steps'] = log_every_n_steps

    start_time = time.time()

    train_step_kwargs['start_time'] = start_time
    train_step_kwargs['max_train_time_sec'] = max_train_time_seconds

    return train_step_kwargs


# Add a train_step_fn to use instead of default
def train_step(sess, train_op, global_step, train_step_kwargs):
    start_time = time.time()

    trace_run_options = None
    run_metadata = None
    if 'should_trace' in train_step_kwargs:
        if 'logdir' not in train_step_kwargs:
            raise ValueError('logdir must be present in train_step_kwargs when '
                             'should_trace is present')
        if sess.run(train_step_kwargs['should_trace']):
            trace_run_options = config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata = config_pb2.RunMetadata()

    total_loss, np_global_step = sess.run([train_op, global_step],
                                          options=trace_run_options,
                                          run_metadata=run_metadata)
    time_elapsed = time.time() - start_time

    if run_metadata is not None:
        tl = timeline.Timeline(run_metadata.step_stats)
        trace = tl.generate_chrome_trace_format()
        trace_filename = os.path.join(train_step_kwargs['logdir'],
                                      'tf_trace-%d.json' % np_global_step)
        logging.info('Writing trace to %s', trace_filename)
        file_io.write_string_to_file(trace_filename, trace)
        if 'summary_writer' in train_step_kwargs:
            train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                                 'run_metadata-%d' %
                                                                 np_global_step)

    total_time_elapsed = (time.time() - train_step_kwargs['start_time'])
    if 'should_log' in train_step_kwargs:
        if np_global_step % train_step_kwargs['log_every_n_steps'] == 0:
            logging.info('global_step: %d\ttotal_time: %.3f\tloss: %.4f\tsec/step: %.3f)',
                         np_global_step, total_time_elapsed, total_loss, time_elapsed)

    if 'max_train_time_sec' in train_step_kwargs and 'start_time' in train_step_kwargs:
        should_stop = total_time_elapsed > train_step_kwargs['max_train_time_sec']
        if should_stop:
            logging.info('stopping after %.3f seconds' % total_time_elapsed)
    else:
        logging.warn('Time Boundaries for training not give. Training will run until stopped')
        should_stop = False

    return total_loss, should_stop


def transfer_learning(root_model_dir, bot_model_dir, protobuf_dir, model_name='inception_v4',
                      dataset_split_name='train',
                      dataset_name='bot',
                      checkpoint_exclude_scopes=None,
                      trainable_scopes=None,
                      max_train_time_sec=None,
                      max_number_of_steps=None,
                      log_every_n_steps=None,
                      save_summaries_secs=None,
                      optimization_params=None):
    """
    :param root_model_dir: Directory containing the root models pretrained checkpoint files
    :param bot_model_dir: Directory where the transfer learned model's checkpoint files are written to
    :param protobuf_dir: Directory for the dataset factory to load the bot's training data from
    :param model_name: name of the network model for the net factory to provide the correct network and preprocesing fn
    :param dataset_split_name: 'train' or 'validation'
    :param dataset_name: triggers the dataset factory to load a bot dataset
    :param checkpoint_exclude_scopes: Layers to exclude when restoring the models variables
    :param trainable_scopes: Layers to train from the restored model
    :param max_train_time_sec: time boundary to stop training after in seconds
    :param max_number_of_steps: maximum number of steps to run
    :param log_every_n_steps:
    :param optimization_params:
    :return: 
    """
    if not optimization_params:
        optimization_params = OPTIMIZATION_PARAMS

    if not max_number_of_steps:
        max_number_of_steps = _MAX_NUMBER_OF_STEPS

    if not checkpoint_exclude_scopes:
        checkpoint_exclude_scopes = _CHECKPOINT_EXCLUDE_SCOPES

    if not trainable_scopes:
        trainable_scopes = _TRAINABLE_SCOPES

    if not max_train_time_sec:
        max_train_time_sec = _MAX_TRAIN_TIME_SECONDS

    if not log_every_n_steps:
        log_every_n_steps = _LOG_EVERY_N_STEPS

    if not save_summaries_secs:
        save_summaries_secs = _SAVE_SUMMARRIES_SECS

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=_NUM_CLONES,
            clone_on_cpu=_CLONE_ON_CPU,
            replica_id=_TASK,
            num_replicas=_WORKER_REPLICAS,
            num_ps_tasks=_NUM_PS_TASKS)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            dataset_name, dataset_split_name, protobuf_dir)

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            model_name,
            num_classes=(dataset.num_classes - _LABELS_OFFSET),
            weight_decay=OPTIMIZATION_PARAMS['weight_decay'],
            is_training=True,
            dropout_keep_prob=OPTIMIZATION_PARAMS['dropout_keep_prob'])

        #####################################
        # Select the preprocessing function #
        #####################################
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            model_name,
            is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=_NUM_READERS,
                common_queue_capacity=20 * _BATCH_SIZE,
                common_queue_min=10 * _BATCH_SIZE)
            [image, label] = provider.get(['image', 'label'])
            label -= _LABELS_OFFSET

            train_image_size = network_fn.default_image_size

            image = image_preprocessing_fn(image, train_image_size, train_image_size)

            images, labels = tf.train.batch(
                [image, label],
                batch_size=_BATCH_SIZE,
                num_threads=_NUM_PREPROCESSING_THREADS,
                capacity=5 * _BATCH_SIZE)
            labels = slim.one_hot_encoding(
                labels, dataset.num_classes - _LABELS_OFFSET)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * deploy_config.num_clones)

        ####################
        # Define the model #
        ####################
        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple clones of network_fn."""
            images, labels = batch_queue.dequeue()
            logits, end_points = network_fn(images)

            #############################
            # Specify the loss function #
            #############################
            if 'AuxLogits' in end_points:
                tf.losses.softmax_cross_entropy(
                    logits=end_points['AuxLogits'], onehot_labels=labels,
                    label_smoothing=_LABEL_SMOOTHING, weights=0.4, scope='aux_loss')
            tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels,
                label_smoothing=_LABEL_SMOOTHING, weights=1.0)
            return end_points

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if OPTIMIZATION_PARAMS['moving_average_decay']:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                OPTIMIZATION_PARAMS['moving_average_decay'], global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
            optimizer = _configure_optimizer(learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if _SYNC_REPLICAS:
            # If sync_replicas is enabled, the averaging will be done in the chief
            # queue runner.
            optimizer = tf.train.SyncReplicasOptimizer(
                opt=optimizer,
                replicas_to_aggregate=_REPLICAS_TO_AGGREGATE,
                variable_averages=variable_averages,
                variables_to_average=moving_average_variables,
                replica_id=tf.constant(_TASK, tf.int32, shape=()),
                total_num_replicas=_WORKER_REPLICAS)
        elif OPTIMIZATION_PARAMS['moving_average_decay']:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = _get_variables_to_train(trainable_scopes)

        #  and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                          name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        ###########################
        # Kicks off the training. #
        ###########################
        slim.learning.train(
            train_tensor,
            logdir=bot_model_dir,
            train_step_fn=train_step,  # Manually added a custom train step to stop after max_time
            train_step_kwargs=_train_step_kwargs(logdir=bot_model_dir, max_train_time_seconds=max_train_time_sec),
            master=_MASTER,
            is_chief=(_TASK == 0),
            init_fn=_get_init_fn(root_model_dir, bot_model_dir, checkpoint_exclude_scopes),
            summary_op=summary_op,
            # number_of_steps=max_number_of_steps,
            log_every_n_steps=log_every_n_steps,
            save_summaries_secs=save_summaries_secs,
            save_interval_secs=_SAVE_INTERNAL_SECS,
            sync_optimizer=optimizer if _SYNC_REPLICAS else None)


def train(bot_model_dir, protobuf_dir, root_model_dir=None, model_name='inception_v4',
          dataset_split_name='train',
          dataset_name='bot',
          max_train_time_sec=None,
          max_number_of_steps=None,
          log_every_n_steps=None,
          save_summaries_secs=None,
          optimization_params=None):
    """

        :param root_model_dir: Directory containing the root models pretrained checkpoint files
        :param bot_model_dir: Directory where the transfer learned model's checkpoint files are written to
        :param protobuf_dir: Directory for the dataset factory to load the bot's training data from
        :param model_name: network model for the net factory to provide the correct network and preprocesing fn
        :param dataset_split_name: 'train' or 'validation'
        :param dataset_name: triggers the dataset factory to load a bot dataset
        :param max_train_time_sec:
        :param max_number_of_steps:
        :parap log_every_n_steps:
        :param OPTIMIZATION_PARAMS: 
        :return: 
    """
    print("""
            INITIALIZING TRAINING OF \t %s\n
            READING PROTOBUF FROM: \t %s \n
            READING MODEL FROM: \t %s \n
            WRITING MODEL TO: \t %s \n
            """
          % (model_name, protobuf_dir, root_model_dir, bot_model_dir))

    transfer_learning(root_model_dir=root_model_dir,
                      bot_model_dir=bot_model_dir,
                      protobuf_dir=protobuf_dir,
                      model_name=model_name,
                      dataset_split_name=dataset_split_name,
                      dataset_name=dataset_name,
                      checkpoint_exclude_scopes=None,
                      trainable_scopes=None,
                      max_train_time_sec=max_train_time_sec,
                      max_number_of_steps=max_number_of_steps,
                      optimization_params=None,
                      save_summaries_secs=save_summaries_secs,
                      log_every_n_steps=log_every_n_steps)


if __name__ == '__main__':
    tf.app.run()
