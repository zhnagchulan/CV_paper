#coding:utf-8


from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.oxflower17 as oxflower17
import tensorflow as tf
import sys


X, Y = oxflower17.load_data(one_hot=True, resize_pics= (227, 227))  ##

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])#输入的x  [batchsize, high, width, channel]
network = conv_2d(network, 96, 11, strides=4, activation='relu', padding='valid')
#network = tf.nn.conv2d(input, filter = tf.Variable(shape = [11, 11, 3, 96]), strides = [1, 4, 4, 1], padding = 'VALID')
#tf.nn.conv2d(input, filter = tf.Variable(shape = [3, 3, 3, 6]), strides, padding, use_cudnn_on_gpu, data_format, dilations, name)
network = max_pool_2d(network, 3, strides=2, padding='valid')
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')  #因为有两个gpu，所以模型chenal*2
network = max_pool_2d(network, 3, strides=2, padding='valid')
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2, padding='valid')
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='adam',#momentum
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=1, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')
#model.predict(X)




"""
输出结果：
---------------------------------
Run id: alexnet_oxflowers17
Log directory: /tmp/tflearn_logs/
---------------------------------
Training samples: 1224
Validation samples: 136
--
Training Step: 1  | time: 6.938s
| Adam | epoch: 001 | loss: 0.00000 - acc: 0.0000 -- iter: 0064/1224
Training Step: 2  | total loss: 2.71017 | time: 11.913s
| Adam | epoch: 001 | loss: 2.71017 - acc: 0.0562 -- iter: 0128/1224
Training Step: 3  | total loss: 7.64756 | time: 16.303s
| Adam | epoch: 001 | loss: 7.64756 - acc: 0.0741 -- iter: 0192/1224
Training Step: 4  | total loss: 7.19106 | time: 20.882s
| Adam | epoch: 001 | loss: 7.19106 - acc: 0.0185 -- iter: 0256/1224
Training Step: 5  | total loss: 7.41119 | time: 25.775s
| Adam | epoch: 001 | loss: 7.41119 - acc: 0.0598 -- iter: 0320/1224
Training Step: 6  | total loss: 7.17491 | time: 30.357s
| Adam | epoch: 001 | loss: 7.17491 - acc: 0.0515 -- iter: 0384/1224
Training Step: 7  | total loss: 6.71597 | time: 35.916s
| Adam | epoch: 001 | loss: 6.71597 - acc: 0.0675 -- iter: 0448/1224
Training Step: 8  | total loss: 7.48070 | time: 40.968s
| Adam | epoch: 001 | loss: 7.48070 - acc: 0.0383 -- iter: 0512/1224
Training Step: 9  | total loss: 7.66680 | time: 45.545s
| Adam | epoch: 001 | loss: 7.66680 - acc: 0.0346 -- iter: 0576/1224
Training Step: 10  | total loss: 6.86781 | time: 50.178s
| Adam | epoch: 001 | loss: 6.86781 - acc: 0.0407 -- iter: 0640/1224
Training Step: 11  | total loss: 6.88267 | time: 54.685s
| Adam | epoch: 001 | loss: 6.88267 - acc: 0.0436 -- iter: 0704/1224
Training Step: 12  | total loss: 6.58990 | time: 60.388s
| Adam | epoch: 001 | loss: 6.58990 - acc: 0.0381 -- iter: 0768/1224
Training Step: 13  | total loss: 5.90852 | time: 69.841s
| Adam | epoch: 001 | loss: 5.90852 - acc: 0.0686 -- iter: 0832/1224
Training Step: 14  | total loss: 5.39155 | time: 78.392s
| Adam | epoch: 001 | loss: 5.39155 - acc: 0.0661 -- iter: 0896/1224
Training Step: 15  | total loss: 5.25769 | time: 84.348s
| Adam | epoch: 001 | loss: 5.25769 - acc: 0.0586 -- iter: 0960/1224
Training Step: 16  | total loss: 4.97955 | time: 90.359s
| Adam | epoch: 001 | loss: 4.97955 - acc: 0.0483 -- iter: 1024/1224
Training Step: 17  | total loss: 5.00766 | time: 96.473s
| Adam | epoch: 001 | loss: 5.00766 - acc: 0.0591 -- iter: 1088/1224
Training Step: 18  | total loss: 4.97198 | time: 101.776s
| Adam | epoch: 001 | loss: 4.97198 - acc: 0.0603 -- iter: 1152/1224
Training Step: 19  | total loss: 4.80236 | time: 107.203s
| Adam | epoch: 001 | loss: 4.80236 - acc: 0.0766 -- iter: 1216/1224
Training Step: 20  | total loss: 5.01538 | time: 110.055s
| Adam | epoch: 001 | loss: 5.01538 - acc: 0.0821 -- iter: 1224/1224
"""