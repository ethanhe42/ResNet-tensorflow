from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import tarfile
import time
from six.moves import urllib
import tensorflow as tf
import google.protobuf

import numpy as np
from data_utils import *
from batch_norm import *
from summary import _activation_summary,_add_loss_summaries

batch_size = 64
initfact=10
lr=.1
path='dataset'
path+='/cifar-100-python'
n_epochs = 80
NUM_CLASSES=20
valid_set=1000
time_per_epoch=10
repeat_layer=2
visual=False


t=time.time()
def elapsed():
    return (time.time()-t)/60

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def eval(xx,yy):
    return str(sess.run(accuracy,
          feed_dict={
              x:xx,
              y:yy,
              is_training: False}))

# %% Setup input to the network and true output label.  These are
# simply placeholders which we'll fill in later.
x = tf.placeholder(tf.float32, [None, 32,32,3])
y = tf.placeholder(tf.float32, [None])

# %% We add a new type of placeholder to denote when we are training.
# This will be used to change the way we compute the network during
# training/testing.
is_training = tf.placeholder(tf.bool, name='is_training')


kernel = _variable_with_weight_decay('conv0',
                                 shape=[3, 3, 3, 16],
                                 stddev=np.sqrt(2.0/initfact/3)
                                 , wd=0.0)
net = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
#net = BatchNorm(net)
net = batch_norm(net,is_training,scope='conv0')
net = tf.nn.relu(net)
if visual:
    _activation_summary(net)
    

for block in range(3):
    nfilters=16<<block
    for layer in range(repeat_layer):
        net_copy=net
        for i in range(2):
            name = 'block_%d/layer_%d/conv%d' % (block, layer,i)
            if block==0:
                i=1
            kernel = _variable_with_weight_decay(name,
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
            net = batch_norm(net,is_training, scope=name)
            #net = BatchNorm(net)
            net = tf.nn.relu(net)
            if visual:
                _activation_summary(net)


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
              strides=[1, 1, 1, 1], 
              padding='VALID',name='global_pooling')
net_shape = net.get_shape().as_list()
net = tf.reshape(net,
        [-1, net_shape[1] * net_shape[2] * net_shape[3]])

weights = _variable_with_weight_decay('softmax_w',
    [64, NUM_CLASSES],
    stddev=1/64.0,
    wd=0.0)
biases = _variable_on_cpu('softmax_b',
    [NUM_CLASSES],
    tf.constant_initializer(0.0))
softmax_linear = tf.add(tf.matmul(net, weights), biases, name='softmax')
y = tf.cast(y, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      softmax_linear, y, name='cross_entropy_per_example')
if visual:
    _activation_summary(cross_entropy)
    summary_op = tf.merge_all_summaries()

cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

# train
train_step=tf.train.MomentumOptimizer(lr,.9).minimize(cross_entropy_mean)
#predict
correct_prediction=tf.equal(tf.argmax(softmax_linear,1),y)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

# load data
Xtr, Ytr, Xte, Yte=load_CIFAR100(path)
mean_image = np.mean(Xtr, axis=0)
Xtr -= mean_image
Xte -= mean_image

numTrain=len(Xtr)-valid_set
iter_per_epoch=numTrain // batch_size

# %% We now create a new session to actually perform the initialization the
# variables:
sess=tf.Session()
sess.run(tf.initialize_all_variables())


if visual:
    summary_writer = tf.train.SummaryWriter("./summary", sess.graph)

for epoch_i in range(n_epochs):
    avg_loss=0
    for batch_i in range(iter_per_epoch):
        idx=np.random.choice(numTrain,batch_size)
        batch_xs=Xtr[idx]
        batch_ys=Ytr[idx]
        loss,_=sess.run([cross_entropy_mean,train_step],
                feed_dict={x: batch_xs, y: batch_ys, is_training: True})
        avg_loss+=loss
        if batch_i%int(iter_per_epoch/10)==0:
            if visual:
                summary_str = sess.run(summary_op,
                        feed_dict={x: batch_xs, y: batch_ys, is_training: False})
                summary_writer.add_summary(summary_str, epoch_i*iter_per_epoch+batch_i)
            print('loss '+str(loss)+',time '+str(elapsed()))

    print("epoch"+str(epoch_i)+
            " avg_loss:"+str(avg_loss/iter_per_epoch)+
            " train acc:"+ eval( batch_xs,batch_ys )+
            " val acc:"+ eval(Xtr[range(-valid_set,-1)],Ytr[range(-valid_set,-1)]))

    if elapsed()>180-time_per_epoch:
        break
print("test acc:"+eval(Xte,Yte))

