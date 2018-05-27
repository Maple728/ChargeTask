import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt

import preprocess

records = pd.read_csv('data/charge.csv')

# ----------- Constant -----------
lr = 200

input_size = 1

timestep_size = 24

hidden_size = 6

layer_num = 2

output_size = 1

batch_size = 5

keep_prob = 0.75

# ------------- test ----------
def get_cycle_time_batch_data_display(records):
    result_y = []
    count = 0
    for hour in range(24):
        hour_data = records[ records['hour'] == hour]
        for i in range(0, len(hour_data)):
            result_y.append(hour_data['power'].iloc[i])
            count += 1
        plt.axvline(count)

    plt.plot(list(range(len(result_y))), result_y, 'r')
    plt.show()

#get_cycle_time_batch_data(records)

class lstm_model():
    def __init__(self, lr, input_size, output_size, hidden_size, layer_num, timesteps, batch_size, keep_prob):
        # build model
        X_ = tf.placeholder(tf.float32, [batch_size, timesteps, input_size])
        
        lstms = [tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size, forget_bias = 1.0, state_is_tuple = True) for i in range(layer_num)]
        drops = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = 1.0, output_keep_prob = keep_prob) for lstm in lstms]
        cell = tf.contrib.rnn.MultiRNNCell(drops, state_is_tuple=True)

        initial_state = cell.zero_state(batch_size, tf.float32)
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, X_, initial_state = initial_state, time_major = False)

        outputs = lstm_outputs[:, -1, :]

        out_W = tf.Variable(tf.truncated_normal([hidden_size, output_size]), dtype = tf.float32)
        out_b = tf.Variable(tf.constant(10.0, shape = [output_size]), dtype = tf.float32)

        y_pre = tf.matmul(outputs, out_W) + out_b

        self.X_ = X_
        self.y_pre = y_pre
        

# --------------- Function -------------------

def get_series_time_batch_data(records, batch_sz, time_step):
    result_x = []
    result_y = []
    
    data_all = list(records['power'])

    for i in range(time_step, len(data_all)):
        result_x.append( data_all[i - time_step : i] )
        result_y.append( data_all[i] )

        if len(result_x) == batch_sz:
            yield np.reshape(result_x, (-1, time_step, input_size)), np.reshape(result_y, (-1, 1))
            # reset data
            result_x = []
            result_y = []
        

def get_cycle_time_batch_data(records, batch_sz, time_step):
    result_x = []
    result_y = []
    for hour in range(24):
        hour_data = records[ records['hour'] == hour]
        for i in range(time_step, len(hour_data)):
            result_x.append(hour_data['power'][i - time_step : i])
            result_y.append(hour_data['power'].iloc[i])

            if len(result_x) == batch_sz:
                yield np.reshape(result_x, (-1, time_step, input_size)), np.reshape(result_y, (-1, 1))
                # reset data
                result_x = []
                result_y = []      

# ---------- LSTM --------

y = tf.placeholder(tf.float32, [batch_size, output_size])

# build lstm layers
lstms = [tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size, forget_bias = 1.0, state_is_tuple = True) for i in range(layer_num)]
drops = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = 1.0, output_keep_prob = keep_prob) for lstm in lstms]
cell = tf.contrib.rnn.MultiRNNCell(drops, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)
lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, X_, initial_state = initial_state, time_major = False)

outputs = lstm_outputs[:, -1, :]

out_W = tf.Variable(tf.truncated_normal([hidden_size, output_size]), dtype = tf.float32)
out_b = tf.Variable(tf.constant(10.0, shape = [output_size]), dtype = tf.float32)

y_pre = tf.matmul(outputs, out_W) + out_b
#y_pre = tf.maximum(y_pre, 0)

loss_func = tf.losses.mean_squared_error(y, y_pre)
train_op = tf.train.AdamOptimizer(lr).minimize(loss_func)

plt.figure(1)
with tf.Session() as sess:
    for i in range(110):
        y_list = []
        y_pre_list = []
        all_loss = []
        sess.run(tf.global_variables_initializer())
        batchs_data = get_series_time_batch_data(records, batch_size, timestep_size)
        for batch in batchs_data:
            # train
            sess.run([train_op], feed_dict = {X_ : batch[0], y : batch[1]})
            predict, loss = sess.run([y_pre, loss_func], feed_dict = {X_ : batch[0], y : batch[1]})
            print('Epoch', i, 'Loss', loss)

            y_list.extend(batch[1])
            y_pre_list.extend(predict)
            all_loss.append(loss)

        x = list(range(len(y_list)))
        if i % 2 == 0:
            plt.subplot(211)
            plt.plot(x, y_list, 'r', x, y_pre_list, 'b')
            plt.subplot(212)
            plt.plot(list(range(len(all_loss))), all_loss, 'g')
            plt.show()

