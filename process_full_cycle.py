from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt

import preprocess

# ----------- Constant -----------
model_path = 'model/model_full_cycle.ckpt'
data_path = 'data/charge.csv'

# ----------- Constant -----------
lr = 0.01

output_size = 1

batch_size = 24

keep_prob = 0.75
cycle_timesteps = 3

is_train = False
        
# model build
X_ = tf.placeholder(tf.float32, [None, cycle_timesteps])
y = tf.placeholder(tf.float32, [None, output_size])

w1_in = tf.Variable(tf.truncated_normal([cycle_timesteps, 12]))
b1_in = tf.Variable(tf.constant(0.1, shape = [12], dtype = tf.float32))

w2_in = tf.Variable(tf.truncated_normal([12, 1]))
b2_in = tf.Variable(tf.constant(0.1, shape = [1], dtype = tf.float32))
 

y_pre = tf.matmul(X_, w1_in) + b1_in
y_pre = tf.matmul(y_pre, w2_in) + b2_in

loss_func = tf.losses.mean_squared_error(y, y_pre)
train_op = tf.train.AdamOptimizer(lr).minimize(loss_func)

# --------- data process ---------
records = pd.read_csv(data_path)
# normalization
scaler = MinMaxScaler()
records['power'] = scaler.fit_transform(records['power'])
    
train_records = records.iloc[: len(records) - 24 * 60]
test_records = records.iloc[len(records) - 24 * 60 : ]

# --------- Function -----------
def get_cycle_time_batch_data(records, batch_sz, time_step):
    cycle_x = []
    y = []

    start_i = 24 * cycle_timesteps
    
    for i in range(start_i, len(records)):
        cur = records.iloc[i]
        cycle_x.append( records[:i][(records['hour'] == cur['hour'])]['power'][ -cycle_timesteps :] )
        y.append(cur['power'])

        if len(y) == batch_sz:
            yield np.reshape(cycle_x, (-1, cycle_timesteps)), np.reshape(y, (-1, output_size))
            # reset data
            series_x = []
            cycle_x = []
            y = []
            
def predict_model(display = False):

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # restore model
        saver.restore(sess, model_path)
        test_data = get_cycle_time_batch_data(records, batch_size, cycle_timesteps)
        test_y_list = [] 
        test_y_pre_list = []
        test_all_loss = []
        for batch in test_data:
            predict, loss = sess.run([y_pre, loss_func], feed_dict = {X_ : batch[0], y : batch[1]})
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
        if display:
            plt.plot(test_x, test_y_list, 'r', test_x, test_y_pre_list, 'b')
            plt.show()

# ----------------- Train ----------
def train():
    plt.figure(1)

    with tf.Session() as sess:
        min_loss = 100000
        saver = tf.train.Saver()
        for i in range(10000):
            y_list = [] 
            y_pre_list = []
            all_loss = []
            sess.run(tf.global_variables_initializer())
            batchs_data = get_cycle_time_batch_data(train_records, batch_size, cycle_timesteps)
            for batch in batchs_data:
                # train
                sess.run([train_op], feed_dict = {X_ : batch[0], y : batch[1]})
                predict, loss = sess.run([y_pre, loss_func], feed_dict = {X_ : batch[0], y : batch[1]})
                

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
                predict_model()

if is_train:
    train()
else:
    predict_model(True)
