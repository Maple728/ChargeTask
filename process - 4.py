from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt

import preprocess

# ----------- Constant -----------
lr = 0.01

input_size = 1

series_timesteps = 12
series_hidden_size = 128
series_layer_num = 2

cycle_timesteps = 6
cycle_hidden_size = 64
cycle_layer_num = 1

output_size = 1

batch_size = 12

keep_prob = 0.75

is_train = False

model_path = 'model/model.ckpt'
# ----------------- Dataset process ----------
records = pd.read_csv('data/charge.csv')
# normalization
scaler = MinMaxScaler()
records['power'] = scaler.fit_transform(records['power'])
train_records = records.iloc[24 * 30: ]
test_records = records.iloc[ : 24 * 30]

class lstm_model():
    def __init__(self, input_size, output_size, hidden_size, layer_num, timesteps, batch_size, keep_prob, name_scope = 'lstm'):
        with tf.name_scope(name_scope):
            # build model
            X_ = tf.placeholder(tf.float32, shape = [None, timesteps, input_size])

            # fully connection input
            w_in = tf.Variable(tf.random_normal([timesteps, timesteps]))
            b_in = tf.Variable(tf.constant(0.1, shape = [timesteps,]))

            #input_ds = tf.reshape(X_, [-1, timesteps])
            #input_rnn = tf.matmul(input_ds, w_in) + b_in
            #input_rnn = tf.reshape(input_rnn, [-1, timesteps, input_size])
            
            
            lstms = [tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size, forget_bias = 1.0, state_is_tuple = True) for i in range(layer_num)]
            drops = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = 1.0, output_keep_prob = keep_prob) for lstm in lstms]
            cell = tf.contrib.rnn.MultiRNNCell(drops, state_is_tuple=True)

            initial_state = cell.zero_state(batch_size, tf.float32)
            lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, X_, initial_state = initial_state, time_major = False, scope = name_scope)

            outputs = lstm_outputs[:, -1, :]
            
            out1_W = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev = 0.1), dtype = tf.float32)
            out1_b = tf.Variable(tf.constant(0., shape = [output_size,]), dtype = tf.float32)
            
            #out2_W = tf.Variable(tf.truncated_normal([timesteps, output_size], stddev = 0.1), dtype = tf.float32)
            #out2_b = tf.Variable(tf.constant(0., shape = [output_size,]), dtype = tf.float32)

            y_pre = tf.matmul(outputs, out1_W) + out1_b
            #y_pre = tf.matmul(y_pre, out2_W) + out2_b

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

# get series and cycle data both
def get_both_batch_data(records, batch_sz, series_timesteps, cycle_timesteps):
    series_x = []
    cycle_x = []
    y = []

    start_i = 24 * cycle_timesteps
    
    for i in range(start_i, len(records)):
        cur = records.iloc[i]
        series_x.append(records['power'][i - series_timesteps : i])
        cycle_x.append( records[:i][(records['hour'] == cur['hour'])]['power'][ -cycle_timesteps :] )
        y.append(cur['power'])

        if len(y) == batch_sz:
            if(len(cycle_x[0]) < 3):
                print(cur)
                print(records[:i][(records['hour'] == cur['hour'])][ -1 - cycle_timesteps :])
            yield np.reshape(series_x, (-1, series_timesteps, input_size)), np.reshape(cycle_x, (-1, cycle_timesteps, input_size)), np.reshape(y, (-1, output_size))
            # reset data
            series_x = []
            cycle_x = []
            y = []


# ---------- Model build --------

y = tf.placeholder(tf.float32, [None, output_size])

# build lstm layers
series_model = lstm_model(input_size = input_size, output_size = output_size, hidden_size = series_hidden_size, layer_num = series_layer_num, timesteps = series_timesteps, batch_size = batch_size, keep_prob = keep_prob, name_scope='series')
cycle_model = lstm_model(input_size = input_size, output_size = output_size, hidden_size = cycle_hidden_size, layer_num = cycle_layer_num, timesteps = cycle_timesteps, batch_size = batch_size, keep_prob = keep_prob, name_scope = 'cycle')

output_W = tf.Variable(tf.truncated_normal([2, 1], stddev = 0.01), dtype = tf.float32)
output_b = tf.Variable(tf.constant(0, dtype = tf.float32))


y_pre = tf.matmul(tf.concat([series_model.y_pre, cycle_model.y_pre], 1), output_W) + output_b

#loss_func = tf.square(y - y_pre)
#loss_func = tf.reduce_mean(tf.where(tf.greater(y_pre, 0), loss_func, loss_func * 20))
loss_func = tf.losses.mean_squared_error(y, y_pre)

train_op = tf.train.AdamOptimizer(lr).minimize(loss_func)


def predict(sess = None):

    if sess is None:
        saver = tf.train.Saver()
        sess = tf.Session()
        # restore model
        saver.restore(sess, model_path)

    test_data = get_both_batch_data(test_records, batch_size, series_timesteps, cycle_timesteps)
    test_y_list = [] 
    test_y_pre_list = []
    test_all_loss = []
    for batch in test_data:
        predict, loss = sess.run([y_pre, loss_func], feed_dict = {series_model.X_ : batch[0], cycle_model.X_ : batch[1], y : batch[2]})
        test_y_list.extend(batch[2])
        test_y_pre_list.extend(predict)
        test_all_loss.append(loss)

    if not is_train:
        # display
        test_x = list(range(len(test_y_list)))

        # inverse normalization
        test_y_list = scaler.inverse_transform(test_y_list)
        test_y_pre_list = scaler.inverse_transform(test_y_pre_list)
        test_y_list = np.array(test_y_list)
        test_y_pre_list = np.array(test_y_pre_list)

        mse = np.mean( (test_y_list - test_y_pre_list) ** 2)
        print('---------------- Test Loss:', np.mean(test_all_loss), 'RMSE:', np.sqrt(mse))
        plt.plot(test_x, test_y_list, 'r', test_x, test_y_pre_list, 'b')
        plt.show()
    else:
        return np.mean(test_all_loss)
        

if not is_train:
    predict()
    exit(1)


with tf.Session() as sess:
    min_loss = 100000
    saver = tf.train.Saver()
    for i in range(100000):
        y_list = [] 
        y_pre_list = []
        all_loss = []
        sess.run(tf.global_variables_initializer())
        batchs_data = get_both_batch_data(train_records, batch_size, series_timesteps, cycle_timesteps)
        for batch in batchs_data:
            # train
            sess.run([train_op], feed_dict = {series_model.X_ : batch[0], cycle_model.X_ : batch[1], y : batch[2]})
            pred, loss = sess.run([y_pre, loss_func], feed_dict = {series_model.X_ : batch[0], cycle_model.X_ : batch[1], y : batch[2]})
            

            y_list.extend(batch[2])
            y_pre_list.extend(pred)
            all_loss.append(loss)

        #m_loss = np.mean(all_loss)
        # TODO: use test loss
        m_loss = predict(sess)
        print('Epoch', i, 'Loss', m_loss)
        
        if m_loss < min_loss: 
            min_loss = m_loss
            print('--------------- Save model:', m_loss)
            # save model
            saver.save(sess, model_path)

            # display
            if False:
                x = list(range(len(y_list)))

                plt.subplot(211)
                plt.plot(x, y_list, 'r', x, y_pre_list, 'b')
                plt.subplot(212)
                plt.plot(list(range(len(all_loss))), all_loss, 'g')
                plt.show()

            # predict
            if False:
                test_data = get_both_batch_data(test_records, batch_size, series_timesteps, cycle_timesteps)
                test_y_list = [] 
                test_y_pre_list = []
                test_all_loss = []
                for batch in test_data:
                    predict, loss = sess.run([y_pre, loss_func], feed_dict = {series_model.X_ : batch[0], cycle_model.X_ : batch[1], y : batch[2]})
                    test_y_list.extend(batch[2])
                    test_y_pre_list.extend(predict)
                    test_all_loss.append(loss)

                # display
                test_x = list(range(len(test_y_list)))

                print('---------------- Test Loss:', np.mean(test_all_loss))
                plt.plot(test_x, scaler.inverse_transform(test_y_list), 'r', test_x, scaler.inverse_transform(test_y_pre_list), 'b')
                plt.show()
