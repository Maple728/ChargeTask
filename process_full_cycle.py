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
lr = 0.001

output_size = 1

batch_size = 24

keep_prob = 0.75
cycle_timesteps = 6

is_train = False
        
# model build
w1_size = 36

X_ = tf.placeholder(tf.float32, [None, cycle_timesteps])
y = tf.placeholder(tf.float32, [None, output_size])

w1_in = tf.Variable(tf.truncated_normal(shape = [cycle_timesteps, w1_size]))
b1_in = tf.Variable(tf.constant(0.1, shape = [w1_size], dtype = tf.float32))

w2_in = tf.Variable(tf.truncated_normal([w1_size, 1]))
b2_in = tf.Variable(tf.constant(0.1, shape = [1], dtype = tf.float32))
 

y_pre = tf.nn.relu(tf.matmul(X_, w1_in) + b1_in)
y_pre = tf.matmul(y_pre, w2_in) + b2_in

loss_func = tf.losses.mean_squared_error(y, y_pre)
train_op = tf.train.AdamOptimizer(lr).minimize(loss_func) 

# --------- data process ---------
records = pd.read_csv(data_path)
# normalization
scaler = MinMaxScaler()
records['power'] = scaler.fit_transform( records['power'].reshape((-1,1))).reshape((-1))
    
train_records = records.iloc[:  - 24 * 30]
test_records = records.iloc[- 24 * 36 : ]

def his_mse(his_records):
    
    test_data = get_cycle_time_batch_data(his_records, 1, cycle_timesteps)

    y_pred_list = []
    y_list = []
    for batch in test_data:
        y_list.extend(scaler.inverse_transform(batch[1]))
        y_pred_list.append(np.mean(scaler.inverse_transform(batch[0])))

    mse = np.mean( (np.array(y_list) - np.array(y_pred_list)) ** 2)

    test_x = list(range(len(y_pred_list)))
    plt.plot(test_x, y_list, 'r', test_x, y_pred_list, 'b')
    plt.show()
    return mse
    

# --------- Function -----------
def get_cycle_time_batch_data(records, batch_sz, time_step, shuffle = False):
    ds = []
    
    start_i = 24 * cycle_timesteps
    
    for i in range(start_i, len(records)):
        cur = records.iloc[i]
        features = list(records[:i][(records['hour'] == cur['hour'])]['power'][ -cycle_timesteps :])
        label = cur['power']
        features.append(label)

        ds.append(features)

    if shuffle:
        np.random.shuffle(ds)

    ds = np.array(ds)
    i = 0
    while i < len(ds):
        end_i = min(i + batch_sz, len(ds))
        yield ds[i : end_i, : -1].reshape( (-1, cycle_timesteps) ), ds[i : end_i, -1].reshape( (-1, 1) )
        i = end_i
            
def predict_model(display = False, sess = None):
    

    # just using test records
    records = test_records
    saver = tf.train.Saver()

    if sess is None:
        sess = tf.Session()
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
    if display:
        print('---------------- His Loss:', his_mse(test_records))
        print('---------------- Test Loss:', np.mean(test_all_loss), 'MSE:', mse)
        plt.plot(test_x, test_y_list, 'r', test_x, test_y_pre_list, 'b')
        plt.show()
    else:
        return mse

# ----------------- Train ----------
def train():
    plt.figure(1)

    with tf.Session() as sess:
        min_test_loss = 1000000000
        
        saver = tf.train.Saver()
        for i in range(1000000000):
            y_list = [] 
            y_pre_list = []
            all_loss = []
            sess.run(tf.global_variables_initializer())
            batchs_data = get_cycle_time_batch_data(train_records, batch_size, cycle_timesteps, shuffle = True)
            for batch in batchs_data:
                # train
                sess.run([train_op], feed_dict = {X_ : batch[0], y : batch[1]})
                predict, loss = sess.run([y_pre, loss_func], feed_dict = {X_ : batch[0], y : batch[1]})
                

                y_list.extend(batch[1])
                y_pre_list.extend(predict)
                all_loss.append(loss)

            if i < 500:
                print('Epoch', i, 'Loss', np.mean(all_loss))
            m_loss = np.mean(all_loss)
            #m_loss = predict_model(sess = sess)
            
            if m_loss < min_test_loss:
                min_test_loss = m_loss
                print('--------------- Save model:', m_loss)
                # save model
                saver.save(sess, model_path)

if is_train:
    train()
else:
    predict_model(True)
