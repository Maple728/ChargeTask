from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt

from preprocess import *

# ----------- Constant -----------
lr = 0.1

input_size = 1

series_timesteps = 3
series_hidden_size = 9
series_layer_num = 1

cycle_timesteps = 6
cycle_hidden_size = 12
cycle_layer_num = 1

output_size = 1

batch_size = 24

keep_prob = 0.75

# ----------------- Dataset process ----------
records = pd.read_csv('data/charge.csv')

class lstm_model():
    def __init__(self, input_size, output_size, hidden_size, layer_num, timesteps, batch_size, keep_prob, name_scope = 'lstm'):
        with tf.name_scope(name_scope):
            # build model
            X_ = tf.placeholder(tf.float32, shape = [None, timesteps, input_size])

            # fully connection input
            w_in = tf.Variable(tf.random_normal([input_size, hidden_size]))
            b_in = tf.Variable(tf.constant(0.1, shape = [hidden_size,]))

            input_ds = tf.reshape(X_, [-1, input_size])
            input_rnn = tf.matmul(input_ds, w_in) + b_in
            input_rnn = tf.reshape(input_rnn, [-1, timesteps, input_size])
            
            
            lstms = [tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size, forget_bias = 1.0, state_is_tuple = True) for i in range(layer_num)]
            drops = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = 1.0, output_keep_prob = keep_prob) for lstm in lstms]
            cell = tf.contrib.rnn.MultiRNNCell(drops, state_is_tuple=True)

            initial_state = cell.zero_state(batch_size, tf.float32)
            lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, X_, initial_state = initial_state, time_major = False, scope = name_scope)

            outputs = lstm_outputs[:, -1, :]
            out_W = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev = 0.1), dtype = tf.float32)
            out_b = tf.Variable(tf.constant(0., shape = [output_size,]), dtype = tf.float32)

            y_pre = tf.matmul(outputs, out_W) + out_b

            self.X_ = X_
            self.y_pre = y_pre

# ---------- Model build --------

def series_train():
    model_path = 'model/model_series.ckpt'
    # build model
    y = tf.placeholder(tf.float32, [None, output_size])

    # build lstm layers
    series_model = lstm_model(input_size = input_size, output_size = output_size, hidden_size = series_hidden_size, layer_num = series_layer_num, timesteps = series_timesteps, batch_size = batch_size, keep_prob = keep_prob, name_scope='series')
    #cycle_model = lstm_model(input_size = input_size, output_size = output_size, hidden_size = cycle_hidden_size, layer_num = cycle_layer_num, timesteps = cycle_timesteps, batch_size = batch_size, keep_prob = keep_prob, name_scope = 'cycle')

    output_W = tf.Variable(tf.truncated_normal([1, 1], stddev = 0.01), dtype = tf.float32)
    output_b = tf.Variable(tf.constant(0, dtype = tf.float32))


    #y_pre = tf.matmul(tf.concat([series_model.y_pre, cycle_model.y_pre], 1), output_W) + output_b
    y_pre = tf.matmul(series_model.y_pre, output_W) + output_b

    loss_func = tf.losses.mean_squared_error(y, y_pre)

    train_op = tf.train.AdamOptimizer(lr).minimize(loss_func)


    # train model
    # normalization
    scaler = MinMaxScaler()
    records['power'] = scaler.fit_transform(records['power'])
        
    train_records = records.iloc[: len(records) - 24 * 60]
    test_records = records.iloc[len(records) - 24 * 60 : ]
    
    # ----------------- Train ----------
    plt.figure(1)

    with tf.Session() as sess:
        min_loss = 100000
        saver = tf.train.Saver()
        for i in range(10000):
            y_list = [] 
            y_pre_list = []
            all_loss = []
            sess.run(tf.global_variables_initializer())
            batchs_data = get_series_time_batch_data(records, batch_size, series_timesteps)
            
            for batch in batchs_data:
                # train
                sess.run([train_op], feed_dict = {series_model.X_ : batch[0], y : batch[1]})
                predict, loss = sess.run([y_pre, loss_func], feed_dict = {series_model.X_ : batch[0], y : batch[1]})
                

                y_list.extend(batch[1])
                y_pre_list.extend(predict)
                all_loss.append(loss)

            m_loss = np.mean(all_loss)
            print('Epoch', i, 'Loss', m_loss)
            
            if m_loss < min_loss:
                min_loss = m_loss
                print('--------------- Save model:', m_loss)
                # save model
                saver.save(sess, model_path)

def predict():
    # normalization
    scaler = MinMaxScaler()
    records['power'] = scaler.fit_transform(records['power'])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # restore model
        saver.restore(sess, model_path)
        test_data = get_cycle_time_batch_data(records, batch_size, cycle_timesteps)
        test_y_list = [] 
        test_y_pre_list = []
        test_all_loss = []
        for batch in test_data:
            predict, loss = sess.run([y_pre, loss_func], feed_dict = {cycle_model.X_ : batch[0], y : batch[1]})
            test_y_list.extend(batch[1])
            test_y_pre_list.extend(predict)
            test_all_loss.append(loss)

        # display
        test_x = list(range(len(test_y_list)))

        # inverse normalization
        test_y_list = scaler.inverse_transform(test_y_list)
        test_y_pre_list = scaler.inverse_transform(test_y_pre_list)
        test_y_list = np.array(test_y_list)
        test_y_pre_list = np.array(test_y_pre_list)

        mse = np.mean( (test_y_list - test_y_pre_list) ** 2)
        print('---------------- Test Loss:', np.mean(test_all_loss), 'MSE:', mse)
        plt.plot(test_x, test_y_list, 'r', test_x, test_y_pre_list, 'b')
        plt.show()
        

if False:
    predict()
    exit(0)

series_train()


