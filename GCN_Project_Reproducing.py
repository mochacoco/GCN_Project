#!/usr/bin/env python
# coding: utf-8

# In[473]:


from scipy import io
import numpy as np
import tensorflow as tf
import math
import h5py
import matplotlib.pyplot as plt
import os
from openpyxl import Workbook
from openpyxl import load_workbook
from scipy.spatial.distance import cdist 
import ot
import keras
from keras.utils import to_categorical
from keras import losses
from keras import optimizers
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.backend import tensorflow_backend as K


fnirs_2 = io.loadmat('fNIRS_data_total_fu_46_active.mat')
Data_2 = fnirs_2['Data2']

#fnirs_3 = io.loadmat('fNIRS_data_total_fu_32_active.mat')
#Data_3 = fnirs_3['Data2']

X = Data_2[0:800,0:56]*1e4
Y = np.reshape(Data_2[0:800,57],[-1,1])-1
Y = to_categorical(Y,2)

Xt = Data_3[0:800,0:56]*1e4
Yt = np.reshape(Data_3[0:800,57],[-1,1])-1
Yt = to_categorical(Yt,2)

X1 = np.zeros((56,320))
X2 = np.zeros((56,320))
c1 = 0
c2 = 0

for i in range(640):
    if Y[i,0] == 1:
        X1[:,c1] = X[i,:].T
        c1 = c1+1
    else:
        X2[:,c2] = X[i,:].T
        c2 = c2+1
R1 = np.matmul(X1,(X1.T))/320
R2 = np.matmul(X2,(X2.T))/320
D, P1 = np.linalg.eig(np.matmul(np.linalg.inv(R2),R1))
w = np.reshape(P1[:,[0,1,54,55]].T,[-1,56])

#Train = np.concatenate([np.matmul(w, X.T).T, Y],1)
#Test = np.concatenate([np.matmul(w, Xt.T).T, Yt],1)
Train = np.concatenate([np.matmul(w, X[0:640,:].T).T, Y[0:640,:]],1)
Test = np.concatenate([np.matmul(w, X[640:800,:].T).T, Y[640:800,:]],1)


# In[474]:


conf = tf.ConfigProto(allow_soft_placement=True)
conf.gpu_options.per_process_gpu_memory_fraction = 0.95
conf.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class fNIRS_CSPGNN(object):
    def __init__(self):
        self.Input_size = 4
        self.Output_size = 2
        self.X= tf.placeholder(tf.float32, [None,self.Input_size,1], name='Placeholder_X')
        self.Y= tf.placeholder(tf.float32, [None,self.Output_size], name='Placeholder_Y')
        #self.DAD = tf.placeholder(tf.float32, [6,6], name='Placeholder_DAD')
        self.DAD = tf.Variable(tf.random_normal([self.Input_size,self.Input_size], stddev=0.03, name='DAD'))
        self.DAD2 = tf.Variable(tf.random_normal([self.Input_size,self.Input_size], stddev=0.03, name='DAD2'))
        self.W1 = tf.Variable(tf.random_normal([1,5],stddev=0.03), name='W1')
        self.W2 = tf.Variable(tf.random_normal([5,3],stddev=0.03), name='W2')
        self.W3 = tf.Variable(tf.random_normal([3,5],stddev=0.03), name='W3')
        
        
        self.H1 = tf.tensordot(self.X, self.W1,axes=[[2],[0]])
        self.O1 = tf.reshape(tf.nn.leaky_relu(tf.tensordot(self.DAD, tf.reshape(self.H1, [self.Input_size,5,-1]), axes=[[1],[0]])),[-1,self.Input_size,5])
        self.H2 = tf.tensordot(self.O1, self.W2,axes=[[2],[0]])
        self.O2 = tf.reshape(tf.nn.leaky_relu(tf.tensordot(self.DAD2, tf.reshape(self.H2, [self.Input_size,3,-1]), axes=[[1],[0]])),[-1,self.Input_size,3])
        #self.H3 = tf.tensordot(self.O2, self.W3,axes=[[2],[0]])
        #self.O3 = tf.reshape(tf.nn.leaky_relu(tf.tensordot(self.DAD, tf.reshape(self.H3, [self.Input_size,5,-1]), axes=[[1],[0]])),[-1,self.Input_size,5])
        
        self.H4 = tf.contrib.layers.flatten(self.O2)
        self.H5 = tf.contrib.layers.fully_connected(self.H4, 20, activation_fn = tf.nn.leaky_relu)
        
        self.Yout = tf.contrib.layers.fully_connected(self.H5, self.Output_size, activation_fn = tf.nn.softmax,
                                                      biases_regularizer = tf.contrib.layers.l2_regularizer(scale=5.),
                                                      weights_initializer = tf.contrib.layers.xavier_initializer())

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Yout, labels=self.Y))
        self.lr = 1e-4
        self.global_step = tf.Variable(0, trainable = False)
        self.decay = tf.train.exponential_decay(self.lr, self.global_step, 100, 0.99, staircase = True)
        self.optimizer = tf.train.AdamOptimizer(self.decay, name='Adam').minimize(self.cost)


# In[475]:


tf.reset_default_graph()
minibatch_size = 128
model = fNIRS_CSPGNN()
Input_size = 4
Output_size =2
Z = Train
Z = np.random.permutation(Z)
X = Z[:,0:Input_size]
Y = np.reshape(Z[:,[Input_size, Input_size+1]], [-1,Output_size])
with tf.Session(config=conf) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(5001):
        Loss = 0
        for i in range((int)((np.shape(X)[0])/minibatch_size)):
            X_batch = X[i*minibatch_size:(i+1)*minibatch_size, :]
            Y_batch = Y[i*minibatch_size:(i+1)*minibatch_size, :]
            X_batch = np.reshape(X_batch, [-1,Input_size,1])
            _,ci = sess.run([model.optimizer, model.cost], feed_dict={model.X:X_batch, model.Y:Y_batch})#, model.DAD:DAD})
            Loss = Loss + ci
        Loss = Loss / (int)(np.shape(X)[0]/minibatch_size)
        if Loss <= 0.35:
            print("Early Stopping","Epoch:",epoch,"Loss:",Loss)
            break
        if epoch % 500 == 0 or epoch <= 3:
            print("Training","Epoch:",epoch,"Loss:",Loss)
    ckpt_dir = './ckpt/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = ckpt_dir + 'fNIRS_GNN' + '.ckpt'
    saver = tf.train.Saver()
    saver.save(sess, ckpt_path)


# In[476]:


import scipy.stats
tf.reset_default_graph()
test_num = 160
Z = Test
X = Z[:,0:Input_size]
Y = np.reshape(Z[:,[Input_size, Input_size+1]], [-1,Output_size])
Prob = np.zeros((Output_size,test_num))
with tf.Session(config=conf) as sess:
    model = fNIRS_CSPGNN()
    ckpt_dir = './ckpt/'
    ckpt_path_location = ckpt_dir + 'fNIRS_GNN' + '.ckpt'
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path_location)
    X = np.reshape(X, [-1,Input_size,1])
    ypred = sess.run([model.Yout], feed_dict={model.X: X})#, model.DAD:DAD})
ypred_ = np.argmax(np.reshape(np.array(ypred),[-1,Output_size]),1)
ypred2 = np.reshape(np.array(ypred),[-1,2])
for i in range(160):
    if ypred2[i,0] <= 0.5 and ypred2[i,1] <= 0.5:
        ypred_[i] = 2
Y_ = np.argmax(Y,1)
score = 0
plt.plot(np.linspace(0,test_num,num=test_num),Y_, c='b')
plt.scatter(np.linspace(0,test_num,num=test_num),ypred_, c='r', s=20*2**(-3))
plt.rcParams["figure.figsize"] = (14,6)

score = 0
for i in range(160):
    if Y_[i] == ypred_[i]:
        score = score + 1
score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




