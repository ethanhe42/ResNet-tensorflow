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

batch_size = 200
initfact=10
learning_rate=.1
path=sys.argv[1]
path+='/cifar-100-python'
n_epochs = 800
NUM_CLASSES=20
valid_set=1000
time_per_epoch=10
repeat_layer=2
visual=True
ifbatchnorm=True
weight_d=0
ifDrop=False
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
    return sess.run(accuracy,
          feed_dict={
              x:xx,
              y:yy,
              is_training: False})

def svd_orthonormal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q
tf.reset_default_graph()
######################architecture##########################################
trainWs=[]
x = tf.placeholder(tf.float32, [None, 32,32,3])
y = tf.placeholder(tf.float32, [None])
is_training = tf.placeholder(tf.bool, name='is_training')

LUSV=tf.placeholder(tf.float32)
lr=tf.placeholder(tf.float32)

kernel = _variable_with_weight_decay('conv0',
                                 shape=[3, 3, 3, 16],
                                 stddev=np.sqrt(2.0/3/9)
                                 , wd=weight_d)
trainWs.append(kernel)
orthoInit0=kernel.assign(svd_orthonormal(kernel.get_shape().as_list()))
upd0=kernel.assign(kernel/LUSV)
ConvLayer0 = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
print(ConvLayer0.get_shape().as_list())
if ifbatchnorm:
    net = batch_norm(ConvLayer0,is_training,scope='conv0')
net = tf.nn.relu(net)
if visual:
    _activation_summary(net)
    
resLayer=[]
orthoInit=[]
kernel_upd=[]
for block in range(3):
    blocks=[]
    nfilters=16<<block
    for layer in range(repeat_layer):
        layers=[]
        net_copy=net
        for i in range(2):
            name = 'block_%d/layer_%d/conv%d' % (block, layer,i)
            if layer==0 and i==0 and block!=0 :
                up=2
            else:
                up=1
            kernel = _variable_with_weight_decay(name,
                    shape=[3, 3,
                           net.get_shape().as_list()[3],
                           nfilters],
                    stddev=np.sqrt(2.0/9/nfilters),
                    wd=weight_d)
            trainWs.append(kernel)
            orthoInit.append(kernel.assign(svd_orthonormal(kernel.get_shape().as_list())))
            kernel_upd.append(kernel.assign(kernel/LUSV))
            
            net  = tf.nn.conv2d(net,
                    kernel,
                    [1,up,up, 1],
                    padding='SAME')
            
            print(net.get_shape().as_list())
            layers.append(net)
            if ifbatchnorm:
                net = batch_norm(net,is_training, scope=name)
            #net = BatchNorm(net)
            net = tf.nn.relu(net)
            if ifDrop:
                net = tf.nn.dropout(net,.5)
            if visual:
                _activation_summary(net)
        blocks.append(layers)

        # residual function (identity shortcut)
        if net_copy.get_shape().as_list()[1]!=net.get_shape().as_list()[1]:
            net_copy=tf.nn.avg_pool(net_copy,[1,2,2,1],
                    strides=[1,2,2,1],padding='VALID')
            net_copy=tf.pad(net_copy,[[0,0],[0,0],[0,0],[0,int(nfilters/2)]])
        net = net + net_copy
    resLayer.append(blocks)

#Global avg pooling
net_shape = net.get_shape().as_list()
net = tf.nn.avg_pool(net,
              ksize=[1, net_shape[1], net_shape[2], 1],
              strides=[1, 1, 1, 1], 
              padding='VALID',name='global_pooling')
net_shape = net.get_shape().as_list()
hid=net_shape[1] * net_shape[2] * net_shape[3]
net = tf.reshape(net,
        [-1, hid])
if ifDrop:
    net = tf.nn.dropout(net,.5)
print(net.get_shape().as_list())
#softmax
weights = _variable_with_weight_decay('softmax_w',
    [hid, NUM_CLASSES],
    stddev=np.sqrt(2.0/hid/NUM_CLASSES),
    wd=weight_d)
biases = _variable_on_cpu('softmax_b',
    [NUM_CLASSES],
    tf.constant_initializer(0.0))

trainWs.append(weights)
trainWs.append(biases)

weights_orth=weights.assign(svd_orthonormal(weights.get_shape().as_list()))
weights_upd=weights.assign(weights/LUSV)

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

########################training########################################
# load data
Xtr, Ytr, Xte, Yte=load_CIFAR100(path)
# simple preprocessing
mean_image = np.mean(Xtr, axis=0)
Xtr -= mean_image
Xte -= mean_image
#img_var=Xtr.std(0)
#Xtr/=img_var
#Xte/=img_var
#M=Xtr.mean(0)
#[D,V]=np.linalg.eig(np.cov(Xtr,rowvar=0))
#
#P = V.dot(np.diag(np.sqrt(1/(D + 0.1)))).dot(V.T)
#Xtr = Xtr.dot(P)
#Xte=Xte.dot(P)

def nextBatch():
   idx=np.random.choice(numTrain,batch_size)
   return Xtr[idx], Ytr[idx]

numTrain=len(Xtr)-valid_set
iter_per_epoch=numTrain // batch_size

# %% We now create a new session to actually perform the initialization the
# variables:
saver = tf.train.Saver()
sess=tf.Session()
sess.run(tf.initialize_all_variables())
#saver.restore(sess,'summary/3_90.ckpt')
if visual:
    summary_writer = tf.train.SummaryWriter("./summary", sess.graph)
#########################LSUV#############################################3
#max_iter = 20;
#needed_variance =.1
#margin = 0.02*needed_variance;
#batch_xs,batch_ys=nextBatch()
#bn=False
#kernel_val,initWeights=sess.run([orthoInit0,ConvLayer0],
#        feed_dict={x: batch_xs, y: batch_ys, is_training: bn})
#for t in range(max_iter):
#    variance=np.var(initWeights)
#    print('var',variance)
#    if abs(variance-needed_variance)<margin:
#        break
#    _,initWeights=sess.run([upd0,ConvLayer0],
#            feed_dict={x: batch_xs, y: batch_ys, is_training: bn,LUSV:np.sqrt(variance/needed_variance)})
#
#for i in range(3):
#    for j in range(repeat_layer):
#        for k in range(2):
#            print('LUSV init',i,j,k)
#            kernel_val,initWeights=sess.run([orthoInit[i*repeat_layer*2+j*2+k],resLayer[i][j][k]],
#                    feed_dict={x: batch_xs, y: batch_ys, is_training: bn})
#            for t in range(max_iter):
#                variance=np.var(initWeights)
#                print(i,j,k,'var',variance)
#                if abs(variance-needed_variance)<margin:
#                    break
#                kernel_val,initWeights=sess.run([kernel_upd[i*repeat_layer*2+j*2+k],resLayer[i][j][k]],
#                        feed_dict={x: batch_xs, y: batch_ys, is_training: bn,LUSV:np.sqrt(variance/needed_variance)})
#
#kernel_val,initWeights=sess.run([weights_orth,cross_entropy],
#        feed_dict={x: batch_xs, y: batch_ys, is_training: bn})
#for t in range(max_iter):
#    variance=np.var(initWeights)
#    print('var',variance)
#    if abs(variance-needed_variance)<margin:
#        break
#    _,initWeights=sess.run([weights_upd,cross_entropy],
#            feed_dict={x: batch_xs, y: batch_ys, is_training: bn,LUSV:np.sqrt(variance/needed_variance)})
########################################################################### 

t=time.time()
losses=[]
valacces=[]
trainacces=[]
best=0
best_val=0

for epoch_i in range(0,n_epochs):
    avg_loss=0
    for batch_i in range(iter_per_epoch):
        batch_xs,batch_ys=nextBatch()   
        loss,_=sess.run([cross_entropy_mean,train_step],
                feed_dict={x: batch_xs, y: batch_ys, is_training: True,lr:learning_rate})
        avg_loss+=loss
        
        if batch_i%int(iter_per_epoch/10)==0:
            losses.append(loss)
            
            if visual:
                summary_str = sess.run(summary_op,
                        feed_dict={x: batch_xs, y: batch_ys, is_training: False})
                summary_writer.add_summary(summary_str, epoch_i*iter_per_epoch+batch_i)
            print('loss '+str(loss)+',time '+str(elapsed()))
    train_acc=eval( batch_xs,batch_ys )
    val_acc=eval(Xtr[-valid_set:],Ytr[-valid_set:])
    print("epoch"+str(epoch_i)+
            " avg_loss:"+str(avg_loss/iter_per_epoch)+
            " train acc:"+ str(train_acc)+
            " val acc:"+ str(val_acc))
    
    valacces.append(val_acc)
    trainacces.append(train_acc)
    if best_val<val_acc:
        best_val=val_acc
        best=epoch_i
    save_path = saver.save(sess,'summary/'+str(repeat_layer)+'_'+ str(epoch_i)+".ckpt")
    print("Model saved in file: %s" % save_path)
        
    if epoch_i>11:
        learning_rate=.01
    if epoch_i>16:
        learning_rate=.005
        if elapsed()>165:
            break

saver.restore(sess,'summary/'+str(repeat_layer)+'_'+str(best)+'.ckpt')
print("test acc:"+str(eval(Xte[:1000],Yte[:1000])))
