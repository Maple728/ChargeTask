from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt

import preprocess

# ----------- Constant -----------
lr = 0.001

input_size = 1
output_size = 1

encoder_timestep = 24
decoder_timestep = 6

series_hidden_size1 = 128
series_hidden_size2 = 32
series_layer_num = 2


batch_size = 100
 
keep_prob = 0.75

is_train = True

model_path = 'model/uber_model.ckpt'

# ----------------- Dataset process ----------
records = pd.read_csv('data/charge.csv')

scaler = MinMaxScaler()
records['power'] = scaler.fit_transform(records['power'])

train_records = records.iloc[ : -24 * 30]
test_records = records.iloc[ -24 * 30 :  ]


# --------------- Function -------------------

def get_series_time_batch_data(records, batch_sz, time_step, shuffle = True):
    ds = []
    
    data_all = list(records['power'])

    for i in range(time_step, len(data_all)):
        row = list(data_all[i - time_step : i])
        row.append(data_all[i])

        ds.append(row)

    if shuffle:
        np.random.shuffle(ds)

    ds = np.array(ds)


    i = 0
    while i < len(ds):
        end_i = i + batch_sz
        if end_i > len(ds):
            break
        
        yield ds[i : end_i, : -1].reshape( (-1, time_step, input_size) ), ds[i : end_i, -1].reshape( (-1, 1) )
        i = end_i

def get_ae_batch_data(records, batch_sz, encode_timestep, decode_timestep, shuffle = True):
    ds = []
    
    data_all = list(records['power'])

    for i in range(encode_timestep + decode_timestep, len(data_all)):
        row = list(data_all[i - (encode_timestep + decode_timestep) : i])

        ds.append(row)

    if shuffle:
        np.random.shuffle(ds)
        
    ds = np.array(ds)

    i = 0
    while i < len(ds):
        end_i = i + batch_sz
        if end_i > len(ds):
            break
        yield ds[i : end_i, : encode_timestep].reshape( (-1, encode_timestep, input_size) ), ds[i : end_i, encode_timestep - decode_timestep : encode_timestep].reshape( (-1, decode_timestep, input_size)), ds[i : end_i, encode_timestep :].reshape( (-1, decode_timestep))
        i = end_i    

def xavier_init(fat_in, fat_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fat_in + fat_out) )
    high = constant * np.sqrt(6.0 / (fat_in + fat_out) )
    return tf.random_uniform( (fat_in, fat_out), minval = low, maxval = high, dtype = tf.float32)


# ---------------- construct models ----------------------
with tf.variable_scope('Encoder_LSTM'):
    encoder_X = tf.placeholder(tf.float32, shape = [None, encoder_timestep, input_size])

    encoder_lstms = [tf.contrib.rnn.BasicLSTMCell(series_hidden_size1), tf.contrib.rnn.BasicLSTMCell(series_hidden_size2)]
    encoder_dropout = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = 1.0, output_keep_prob = keep_prob) for lstm in encoder_lstms]
    encoder_cells = tf.contrib.rnn.MultiRNNCell(encoder_dropout, state_is_tuple = True)

    encoder_initial_state = encoder_cells.zero_state(batch_size, tf.float32)
    encoder_all_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cells, encoder_X, initial_state = encoder_initial_state)

    encode_outputs = encoder_all_outputs[:, -1, :]

with tf.variable_scope('Decoder_LSTM'):
    decoder_X = tf.placeholder(tf.float32, shape = [None, decoder_timestep, input_size])
    decoder_Y = tf.placeholder(tf.float32, shape = [None, decoder_timestep])

    decoder_lstms = [tf.contrib.rnn.BasicLSTMCell(series_hidden_size1), tf.contrib.rnn.BasicLSTMCell(series_hidden_size2)]
    decoder_dropout = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = 1.0, output_keep_prob = keep_prob) for lstm in decoder_lstms]
    decoder_cells = tf.contrib.rnn.MultiRNNCell(decoder_dropout, state_is_tuple = True)

    decoder_initial_state = encoder_final_state
    decoder_all_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cells, decoder_X, initial_state = decoder_initial_state)

    decoder_out_W = tf.Variable(tf.truncated_normal(shape = [series_hidden_size2 * decoder_timestep, decoder_timestep]))
    decoder_out_b = tf.Variable(tf.constant(0.0, shape = [decoder_timestep], dtype = tf.float32))

    decoder_outputs_flat = tf.reshape(decoder_all_outputs, [-1, series_hidden_size2 * decoder_timestep])

    decoder_losses = tf.matmul(decoder_outputs_flat, decoder_out_W) + decoder_out_b

    decoder_losses = tf.losses.mean_squared_error(decoder_losses, decoder_Y)
    

    decoder_train_op = tf.train.AdamOptimizer(lr).minimize(decoder_losses)

with tf.variable_scope('Inference'):
    Y = tf.placeholder(tf.float32, shape = [None, 1])

    inference_inputs = tf.reshape(encode_outputs, [-1, series_hidden_size2])
    
    inference_fc_w1 = tf.Variable(xavier_init(series_hidden_size2, 64))
    inference_fc_b1 = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32))

    inference_fc_w2 = tf.Variable(xavier_init(64, 16))
    inference_fc_b2 = tf.Variable(tf.constant(0.0, shape = [16], dtype = tf.float32))

    inference_fc_w3 = tf.Variable(xavier_init(16, 1))
    inference_fc_b3 = tf.Variable(tf.constant(0.0, shape = [1], dtype = tf.float32))    
    
    y_pre1 = tf.nn.tanh( tf.matmul(inference_inputs, inference_fc_w1) + inference_fc_b1 )
    y_pre2 = tf.nn.tanh( tf.matmul(y_pre1, inference_fc_w2) + inference_fc_b2 )
    inference_y_pre = tf.matmul(y_pre2, inference_fc_w3) + inference_fc_b3

    inference_losses = tf.losses.mean_squared_error(inference_y_pre, Y)
    inference_train_op = tf.train.AdamOptimizer(lr).minimize(inference_losses)

# session saver
saver = tf.train.Saver()

# --------- train model ------------------------------
def predict_model(target_records, sess = None, display = True):
    # sess process
    if sess is None:
        sess = tf.Session()
        saver.restore(sess, model_path)
    
    # test
    test_series_batch_data = get_series_time_batch_data(target_records, batch_size, encoder_timestep, shuffle = False)

    y_list = []
    y_pre_list = []
    all_loss = []
    for batch_data in test_series_batch_data:
        loss, pred = sess.run([inference_losses, inference_y_pre], feed_dict = {encoder_X : batch_data[0], Y : batch_data[1]})
        all_loss.append(loss)

        y_list.extend(batch_data[1])
        y_pre_list.extend(pred)

    # display
    x_list = list(range(len(y_list)))

    # inverse normalization
    y_list = np.array(scaler.inverse_transform(y_list))
    y_pre_list = np.array(scaler.inverse_transform(y_pre_list))

    rmse = np.sqrt(np.mean( (y_pre_list - y_list) ** 2))
    print('---------------- Inference Test Loss:', np.mean(all_loss), 'RMSE:', rmse)
    
    if display:
        plt.plot(x_list, y_list, 'r', x_list, y_pre_list, 'b')
        plt.show()    
    

def train_model():
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        
        # train ae
        for i in range(300):
            ae_batch_data = get_ae_batch_data(train_records, batch_size, encoder_timestep, decoder_timestep)
            all_loss = []
            for batch_data in ae_batch_data:
                loss, _ = sess.run([decoder_losses, decoder_train_op], feed_dict = {encoder_X : batch_data[0], decoder_X : batch_data[1], decoder_Y : batch_data[2]})
                all_loss.append(loss)

            print('Encode-Decode train Epoch', i, 'Loss:', np.mean(all_loss))

        # train inference
        for i in range(2000):
            series_batch_data = get_series_time_batch_data(train_records, batch_size, encoder_timestep)
            all_loss = []
            for batch_data in series_batch_data:
                loss, _ = sess.run([inference_losses, inference_train_op], feed_dict = {encoder_X : batch_data[0], Y : batch_data[1]})
                all_loss.append(loss)

            print('Inference train epoch', i, 'Loss:', np.mean(all_loss))

        saver.save(sess, model_path)
        predict_model(train_records, sess)
        predict_model(test_records, sess)

if is_train:
    train_model()
else:
    predict_model(train_records, display = True)
    predict_model(test_records)


