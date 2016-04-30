from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import namedtuple
from math import sqrt

from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import skflow

from data_utils import load_CIFAR100

def weight_variable(shape,name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name)

def bias_variable(shape,name=None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def res_net(x, y, activation=tf.nn.elu):

    with tf.variable_scope('conv_layer1'):
        net = tf.nn.conv2d(x, 16, [3, 3], batch_norm=True,
                                activation=activation, bias=False,
                                padding='SAME')

    for block in range(3):
        nfilters=16<<block
        for layer in range(1):
            net_copy=net
            name = 'block_%d/layer_%d' % (block, layer)
            for i in range(2):
                with tf.variable_scope(name+'/'+str(i)):
                    if block==0:
                        i=1
                    net = skflow.ops.conv2d(net,
                                nfilters,
                                [3, 3], [1, 2-i, 2-i, 1],
                                padding='SAME',
                                activation=activation,
                                batch_norm=True,
                                bias=False)

            # residual function (identity shortcut)
            if net_copy.get_shape().as_list()[1]!=net.get_shape().as_list()[1]:
                net_copy=tf.nn.avg_pool(net_copy,[1,2,2,1],
                        strides=[1,2,2,1],padding='VALID')
                net_copy=tf.pad(net_copy,[[0,0],[0,0],[0,0],[0,int(nfilters/2)]])
            net = net + net_copy

    #Global avg pooling
    net_shape = net.get_shape().as_list()
    net = tf.nn.avg_pool(net,
                  ksize=[1, net_shape[1], net_shape[2], 1],
                  strides=[1, 1, 1, 1], padding='VALID')
    net_shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

    w=tf.get_variable('w',[net.get_shape()[1],y.get_shape()[-1]])
    b=tf.get_variable('b',[y.get_shape()[-1]])
    logits = tf.nn.xw_plus_b(net, w, b)
    h = tf.nn.softmax_cross_entropy_with_logits(logits, y, name='h_raw')
    loss = tf.reduce_mean(h, name='cross_entropy')
    predictions = tf.nn.softmax(logits, name=name)
    
   #return skflow.models.logistic_regression(net,y)
    return predictions, loss

path='./dataset/cifar-100-python'
Xtr, Ytr, Xte, Yte=load_CIFAR100(path)
nclass=20
w=weight_variable([64,nclass],'w')
b=bias_variable([nclass,1],'b')
classifier = skflow.TensorFlowEstimator(
     model_fn=res_net, 
     n_classes=nclass, batch_size=128, steps=100,
     learning_rate=0.1, continue_training=True,
     optimizer="Adam")

while True:
    # Train model and save summaries into logdir.
    classifier.fit(Xtr, Ytr, logdir="models/resnet/")

    # Calculate accuracy.
    score = metrics.accuracy_score(
        Yte, classifier.predict(Xtr, batch_size=64))
    print('Accuracy: {0:f}'.format(score))

    # Save model graph and checkpoints.
    classifier.save("models/resnet/")
