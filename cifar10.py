# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10_input
import numpy as np

BATCH_SIZE=256
DATA_DIR='/tmp/cifar10_data'
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not DATA_DIR:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=BATCH_SIZE)


def inputs(eval_data):
  if not DATA_DIR:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=BATCH_SIZE)


def BatchNorm(x, use_local_stat=True, decay=0.9, epsilon=1e-5):
    shape = x.get_shape().as_list()
    assert len(shape) in [2, 4]

    n_out = shape[-1]  # channel
    beta = tf.get_variable('beta', [n_out])
    gamma = tf.get_variable(
        'gamma', [n_out],
        initializer=tf.constant_initializer(1.0))

    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0], keep_dims=False)
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=False)

    ema = tf.train.ExponentialMovingAverage(decay=decay)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    if use_local_stat:
        with tf.control_dependencies([ema_apply_op]):
            return tf.nn.batch_normalization(
                x, batch_mean, batch_var, beta, gamma, epsilon, 'bn')
    else:
        batch = tf.cast(tf.shape(x)[0], tf.float32)
        mean, var = ema_mean, ema_var * batch / (batch - 1) # unbiased variance estimator
        return tf.nn.batch_normalization(
            x, mean, var, beta, gamma, epsilon, 'bn')


def inference(images):
    # conv1
    initfact=10
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 16],
                                         stddev=np.sqrt(2.0/initfact/3)
                                         , wd=0.0)
        net = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        net = BatchNorm(net)
        net = tf.nn.elu(net, name=scope.name)
        _activation_summary(net)
        
    
    for block in range(3):
        nfilters=16<<block
        for layer in range(1):
            net_copy=net
            name = 'block_%d/layer_%d' % (block, layer)
            for i in range(2):
                with tf.variable_scope(name+'/'+str(i)):
                    if block==0:
                        i=1
                    kernel = _variable_with_weight_decay('weights',
                            shape=[3, 3,
                                   net.get_shape().as_list()[3],
                                   nfilters],
                            stddev=np.sqrt(2.0/initfact/nfilters),
                            wd=0.0)
                    if layer==0 and block!=0 and i==0:
                        up=1
                    else:
                        up=0
                    net  = tf.nn.conv2d(net,
                            kernel,
                            [1,1+up,1+up, 1],
                            padding='SAME')
                    net = BatchNorm(net)
                    net = tf.nn.elu(net, name=scope.name)
                    _activation_summary(net)
    

       # residual function (identity shortcut)
            if net_copy.get_shape().as_list()[1]!=net.get_shape().as_list()[1]:
                net_copy=tf.nn.avg_pool(net_copy,[1,2,2,1],
                        strides=[1,2,2,1],padding='VALID')
                net_copy=tf.pad(net_copy,[[0,0],[0,0],[0,0],[0,int(nfilters/2)]])
            net = net + net_copy

    with tf.variable_scope('global_pool') as scope: 
        #Global avg pooling
        net_shape = net.get_shape().as_list()
        net = tf.nn.avg_pool(net,
                      ksize=[1, net_shape[1], net_shape[2], 1],
                      strides=[1, 1, 1, 1], 
                      padding='VALID')
        net_shape = net.get_shape().as_list()
        net = tf.reshape(net,
                [-1, net_shape[1] * net_shape[2] * net_shape[3]])

  # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights',
            [64, NUM_CLASSES],
            stddev=1/64.0,
            wd=0.0)
        biases = _variable_on_cpu('biases',
            [NUM_CLASSES],
            tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(net, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear



def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)
  
  lr=.1

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.MomentumOptimizer(lr,.9)

    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def accuracy(logits,labels):
    
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    variable_averages=tf.train.Exponential
    ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY)
    avg_grad = ema.average_name(grads)
    saver = tf.train.Saver({avg_grad: grads})
    
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
      threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                       start=True))
    
    num_iter = int(math.ceil(10000 / 256))
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = num_iter * 256
    step = 0
    while step < num_iter and not coord.should_stop():
      predictions = sess.run([top_k_op])
      true_count += np.sum(predictions)
      step += 1
    
    # Compute precision @ 1.
    precision = true_count / total_sample_count
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    
    summary = tf.Summary()
    summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='Precision @ 1', simple_value=precision)
    summary_writer.add_summary(summary, global_step)
    
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)




def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = DATA_DIR
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
