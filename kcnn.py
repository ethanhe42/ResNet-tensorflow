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
from kmeans import kmeans,extract_features
batch_size = 64
initfact=10
learning_rate=.1
path='dataset'
path+='/cifar-100-python'
n_epochs = 800
NUM_CLASSES=20
valid_set=1000
time_per_epoch=10
repeat_layer=1
visual=False
ifbatchnorm=True
weight_d=0.001
ifDrop=False
features=256
def elapsed():
    return (time.time()-t)/60

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None and wd !=0:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def eval(xx,yy):
    return str(sess.run(accuracy,
          feed_dict={
              x:xx,
              y:yy,
              is_training: False}))

def svd_orthonormal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q
######################architecture##########################################
trainWs=[]
x = tf.placeholder(tf.float32, [None,9,9,features])
y = tf.placeholder(tf.float32, [None])
is_training = tf.placeholder(tf.bool, name='is_training')

LUSV=tf.placeholder(tf.float32)
lr=tf.placeholder(tf.float32)

kernel = _variable_with_weight_decay('conv0',
                                 shape=[3, 3, features, features],
                                 stddev=np.sqrt(2.0/initfact/3)
                                 , wd=weight_d)
ConvLayer0 = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
net = tf.nn.relu(ConvLayer0)
if visual:
    _activation_summary(net)
    

#Global avg pooling
net_shape = net.get_shape().as_list()
print(net_shape)
net = tf.nn.avg_pool(net,
              ksize=[1, 3, 3, 1],
              strides=[1, 3, 3, 1], 
              padding='VALID',name='global_pooling')

net_shape = net.get_shape().as_list()
hidden_inp=9*net_shape[1] * net_shape[2] * net_shape[3]
print(net_shape)
n_fc = 400
net = tf.reshape(net,
        [-1, hidden_inp])

hw = _variable_with_weight_decay('hw',
    [hidden_inp,n_fc],
    stddev=1/hidden_inp,
    wd=weight_d)
hb = _variable_on_cpu('hb',
    [n_fc],
    tf.constant_initializer(0.0))

# %% Create a fully-connected layer:
net = tf.nn.relu(tf.matmul(net, hw) + hb)


#softmax
weights = _variable_with_weight_decay('softmax_w',
    [n_fc, NUM_CLASSES],
    stddev=1/n_fc,
    wd=weight_d)
biases = _variable_on_cpu('softmax_b',
    [NUM_CLASSES],
    tf.constant_initializer(0.0))

trainWs.append(weights)
trainWs.append(biases)


softmax_linear = tf.add(tf.matmul(net, weights), biases, name='softmax')

y = tf.cast(y, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      softmax_linear, y, name='cross_entropy_per_example')
if visual:
    _activation_summary(cross_entropy)
    summary_op = tf.merge_all_summaries()

cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
tf.add_to_collection('losses', cross_entropy_mean)
cross_entropy_mean=tf.add_n(tf.get_collection('losses'))
# train
#train_step=tf.train.MomentumOptimizer(lr,.9).minimize(cross_entropy_mean)
train_step=tf.train.AdagradOptimizer(lr).minimize(cross_entropy_mean)

#predict
correct_prediction=tf.equal(tf.argmax(softmax_linear,1),y)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
tess=[cross_entropy_mean,train_step]
#########################preprocessing#######################################
t=time.time()
# load data
Xtr, Ytr, Xte, Yte=load_CIFAR100(path)
# simple preprocessing
mean_image = np.mean(Xtr, axis=0)
Xtr -= mean_image
Xte -= mean_image
Xtr=Xtr.swapaxes(1,3)
Xte=Xte.swapaxes(1,3)

centroids,a,b,c,d=kmeans(Xtr,1600,selected_feats=features)
print(elapsed())
Xtr=extract_features(Xtr,centroids,a,b,c,d)
Xte=extract_features(Xte,centroids,a,b,c,d)
print(elapsed())

##########################training###############################

def nextBatch():
   idx=np.random.choice(numTrain,batch_size)
   return Xtr[idx], Ytr[idx]

numTrain=len(Xtr)-valid_set
iter_per_epoch=numTrain // batch_size

# %% We now create a new session to actually perform the initialization the
# variables:
saver = tf.train.Saver()
sess=tf.Session()
#saver.restore(sess,'26.ckpt')
sess.run(tf.initialize_all_variables())
if visual:
    summary_writer = tf.train.SummaryWriter("./summary", sess.graph)



for epoch_i in range(n_epochs):
    avg_loss=0
    for batch_i in range(iter_per_epoch):
        batch_xs,batch_ys=nextBatch()   
        loss,_=sess.run([cross_entropy_mean,train_step],
                feed_dict={x: batch_xs, y: batch_ys, is_training: True,lr:learning_rate})
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
    save_path = saver.save(sess,'summary/'+str(repeat_layer)+'_'+ str(epoch_i)+".ckpt")
    print("Model saved in file: %s" % save_path)
    for w in trainWs:
        a=w.eval(session=sess)
        print(a.shape,a.mean(),a.std())
    if epoch_i>83:
        learning_rate=.01
    if epoch_i>125:
        learning_rate=.001
    if epoch_i>162:
        break
#    if elapsed()>180-time_per_epoch:
#        break
print("test acc:"+eval(Xte,Yte))

