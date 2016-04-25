import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread

def load_CIFAR_batch(filename, num):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['coarse_labels']
    X = X.reshape(num, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR100(ROOT):
  """ load all of cifar """
  Xtr, Ytr = load_CIFAR_batch(os.path.join(ROOT, 'train'),50000)
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test'),10000)
  return Xtr, Ytr, Xte, Yte
