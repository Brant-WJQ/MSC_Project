#obtain project root path
import sys
import matplotlib.pyplot as plt
root_path = sys.path[0]
sys.path.append(root_path)
import tensorflow as tf
import numpy as np
import pandas as pd
from function.cnn_RandomGradient import netural_network
from function.process_data import get_data
# from ..Recognize_Protain.hanshu.cnn_RandomGradient import netural_network
############

trainX, trainY, testX, testY = get_data(filename='new.csv')

'''AE'''
num_feature = 50 #the number of feature extracted
nn = netural_network(inputs_num=trainX.shape[1],
                         outputs_num=trainX.shape[1],
                         hidden_nums=[num_feature],
                         learn_rate=10**(-2),
                         epoch_times=500,
                         SAE_alph=0,
                         forward_type='tanh',
                         log_dir=root_path+'/logs/FSnum_'+str(num_feature)+'/')

#training
final_params = nn.train(trainX,trainX,batchsize=1)
#output of encoding
with tf.Session() as sess:
    encoder_test = nn.hiddenLayer_computation(testX,final_params,output_layer=1)
    encoder_train = nn.hiddenLayer_computation(trainX, final_params, output_layer=1)
    sess.run(tf.global_variables_initializer())
    encoder_test,encoder_train = sess.run([encoder_test,encoder_train])  #result
    sess.close()

del nn

'''three layers of FCL'''
# Xtrain = encoder_train
# Ytrain = trainY
# Xtest = encoder_test
# Ytest = testY

nn = netural_network(inputs_num=encoder_train.shape[1],
                     outputs_num=trainY.shape[1],
                     hidden_nums=[100,100],
                     learn_rate=10 ** (-2),
                     epoch_times=2000,
                     SAE_alph=0,
                     forward_type='tanh',
                     log_dir=root_path + '/logs/FSnum_3layersConnect_' + str(num_feature) + '/',
                     lossFunction='no_cross_entropy')

# training
final_params = nn.train(encoder_train, trainY, batchsize=1)
# output of encoding
with tf.Session() as sess:
    testY_predict = nn.forward_computation(encoder_test, final_params,train=False)
    sess.run(tf.global_variables_initializer())
    testY_predict = sess.run(testY_predict)  # result
    sess.close()

del nn


'''calculate accuracy'''
testY_predict_class = np.argmax(testY_predict,axis=1)
testY_class = np.argmax(testY,axis=1)
#
accurancy = 1 - np.sum((testY_predict_class - testY_class)**2)/len(testY_class)
output = np.zeros([testY_predict.shape[0],6])
output[:,0:2] = testY_predict
output[:,2] = testY_predict_class
output[:,3:5] = testY
output[:,5] = testY_class
output = pd.DataFrame(output,columns=['fit0','fit1','fit_class','t0','t1','t_class'])
#
print(output)
print(accurancy)
