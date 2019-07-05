import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import matplotlib.pyplot as plt
#obtain project root path
'''
import sys
root_path = sys.path[0]
sys.path.append(root_path)
import function.demo_tensorboard as show_tb
'''
import demo_tensorboard as show_tb

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

class netural_network:


    def __init__(self,
                 inputs_num=45,
                 outputs_num=45,
                 hidden_nums=[200,200,200],
                 learn_rate=0.01,
                 epoch_times=1000,
                 SAE_alph=0,
                 forward_type='sig',
                 log_dir = '/logs/',
                 lossFunction = 'no_cross_entropy'):


        show_tb.open_tensorboard(log_dir)
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)
        self.sess = tf.Session()

        self.L1_alph = SAE_alph
        self.learn_rate = learn_rate
        self.epoch_times = epoch_times
        self.layer_nums = [inputs_num] + hidden_nums + [outputs_num]
        self.forward_type = forward_type
        self.log_dir = log_dir
        self.lossFunction = lossFunction

    def __del__(self):
        self.sess.close()
        tf.reset_default_graph()
        print('del have been done')

    def init_params(self):

        layer_nums = self.layer_nums
        print('--------------initialize network----------------')
        print('number of layers：',len(layer_nums))
        print('number of neurons in each layer：',layer_nums)
        print('---------------------------------------')
        print('---------------------------------------')


        params_W = []
        params_Bias = []
        for i in range(len(layer_nums)-1):
                ###
            W_name = 'W'+str(i+1)
            bias_name = 'B'+str(i+1)
            with tf.name_scope('init_params'):

                params_W.append(
                    tf.get_variable(shape=[layer_nums[i], layer_nums[i+1]],initializer=layers.xavier_initializer(seed=1),name=W_name,dtype=tf.float64)
                )
                ###
                params_Bias.append(
                    tf.get_variable(shape=[1,layer_nums[i+1]],name=bias_name,dtype=tf.float64,initializer=layers.xavier_initializer(seed=1))
                )
        return [params_W,params_Bias]


    def forward_computation(self,inputsX,params,train=True):
        params_W, params_Bias = params
        type = self.forward_type
        with tf.name_scope('forward_computing'):
            # sig forward calculation
            if type == 'sig':
                temp1 = tf.sigmoid(tf.matmul(inputsX, params_W[0]) + params_Bias[0])
                for i in range(len(params_W)-2):
                    temp1 = tf.sigmoid(tf.matmul(temp1,params_W[i+1]) + params_Bias[i+1])
                temp1 = tf.matmul(temp1, params_W[-1]) + params_Bias[-1]
            # tanh forward calculation
            if type == 'tanh':
                temp1 = tf.tanh(tf.matmul(inputsX, params_W[0]) + params_Bias[0])
                for i in range(len(params_W) - 2):
                    temp1 = tf.tanh(tf.matmul(temp1, params_W[i + 1]) + params_Bias[i + 1])
                temp1 = tf.matmul(temp1, params_W[-1]) + params_Bias[-1]
            # linear forward calculation
            if type == 'normal':
                temp1 = tf.matmul(inputsX, params_W[0]) + params_Bias[0]
                for i in range(len(params_W) - 1):
                    temp1 = tf.matmul(temp1, params_W[i + 1]) + params_Bias[i + 1]

            # ReLu forward calculation
            if type == 'relu':
                temp1 = tf.nn.relu(tf.matmul(inputsX, params_W[0]) + params_Bias[0])
                for i in range(len(params_W) - 1):
                    temp1 = tf.nn.relu(tf.matmul(temp1, params_W[i + 1]) + params_Bias[i + 1])

        # dropout
        if train: temp1 = tf.nn.dropout(temp1,1)
        return temp1

    def hiddenLayer_computation(self, inputsX, params,output_layer=1):
        params_W, params_Bias = params
        type = self.forward_type
        with tf.name_scope('forward_computing'):
            # sig forward calculation
            if type == 'sig':
                temp1 = tf.sigmoid(tf.matmul(inputsX, params_W[0]) + params_Bias[0])
                for i in range(output_layer-1):
                    temp1 = tf.sigmoid(tf.matmul(temp1, params_W[i + 1]) + params_Bias[i + 1])
            # tanh forward calculation
            if type == 'tanh':
                temp1 = tf.tanh(tf.matmul(inputsX, params_W[0]) + params_Bias[0])
                for i in range(output_layer-1):
                    temp1 = tf.tanh(tf.matmul(temp1, params_W[i + 1]) + params_Bias[i + 1])
            # linear forward calculation
            if type == 'normal':
                temp1 = tf.matmul(inputsX, params_W[0]) + params_Bias[0]
                for i in range(output_layer-1):
                    temp1 = tf.matmul(temp1, params_W[i + 1]) + params_Bias[i + 1]

        return temp1

    def cal_loss(self,y_fit,y_true,params):
        with tf.name_scope('loss'):
            alph = 0
            #calculate L1 regulation
            params_W = params[0]
            L1_W_loss = 0
            for i in params_W : L1_W_loss += tf.reduce_sum(tf.sqrt(i**2))
            L1_W_loss = L1_W_loss/len(params_W)
            #calculate loss value
            loss_MSE = tf.reduce_mean(tf.sqrt((y_fit - y_true)**2)) + L1_W_loss*alph
            #entropy
            clip_yfit = tf.clip_by_value(tf.sigmoid(y_fit), 1e-3, 1.0)
            loss_cross_entropy = -tf.reduce_mean( y_true * tf.log(clip_yfit) + (1-y_true)*tf.log(1-clip_yfit) )

        tf.summary.scalar('Mean_Square_Error',loss_MSE)

        if self.lossFunction == 'cross_entropy':
            return loss_cross_entropy
        else:
            return loss_MSE

    def train(self,inputX,inputY,batchsize=1):
        #batchsize

        params = self.init_params()
        #
        trainX = tf.placeholder(dtype=inputX.dtype,name='inputsX')
        trainY = tf.placeholder(dtype=inputY.dtype,name='true_Y')
        #
        y_fit = self.forward_computation(trainX, params)
        loss = self.cal_loss(y_fit, trainY, params)
        opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(loss)
        #
        self.sess.run(tf.global_variables_initializer())
        # merged
        merged = tf.summary.merge_all()
        ##initialize
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        #
        min_loss_params = 0
        min_loss = 10 ** 8

        for ii in range(batchsize*2):
            #random grouping
            if batchsize >1:
                indexRandom = np.random.randint(0,inputX.shape[0]-1,int(inputX.shape[0]/batchsize))
                feed_X = inputX[indexRandom,:]
                feed_Y = inputY[indexRandom,:]
            if batchsize ==1:
                feed_X = inputX
                feed_Y = inputY
            for i in range(self.epoch_times):
                # 逆误差回馈
                iloss,iy_fit,kk = self.sess.run([loss,y_fit,opt],feed_dict={trainX:feed_X,trainY:feed_Y})
                # 保存最低loss
                # print(iloss)
                if iloss < min_loss:
                    min_loss = iloss
                    min_loss_params = self.sess.run(params)
                # 过程输出
                if i%10 ==0:
                    #save tensorboard data
                    sumarys = self.sess.run(merged,feed_dict={trainX:feed_X,trainY:feed_Y})
                    writer.add_summary(sumarys, i+ii*self.epoch_times)

                    # ax.plot(feed_X[:,0], iy_fit[:,0], 'r.')
                    # plt.pause(0.0000000000000000000001)
                    # ax.lines.pop(1)

                    txt = 'epoch times:%d, Loss:%f, min_loss:%f.'%(i+ii*self.epoch_times,iloss,min_loss)
                    print(txt)

        return_params = self.sess.run(params)
        #

        writer.close()
        # return min_loss_params
        return return_params





if __name__ == '__main__':



    x = np.random.rand(2000,1)*10-5
    y = (x-0.2)**2
    # y = np.sin(x)
    ax.plot(x[:,0], y[:,0], 'b.')
    plt.ylim([-1,1])

    nn = netural_network(inputs_num=x.shape[1],
                         outputs_num=y.shape[1],
                         hidden_nums=[50,50],
                         learn_rate=10**(-2),
                         epoch_times=40000,
                         #L1_alph=0,
                         forward_type='normal')

    #train
    final_params = nn.train(x,y,batchsize=10)

    #validation

    yf = nn.forward_computation(x,final_params)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    yf = sess.run(yf)
    #







