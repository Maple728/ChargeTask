from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt

# ----------- Constant -----------
dropout_sizes_threhold = 10

input_size = 1
output_size = 1
batch_size = 100

class SeriesConfig:
    lr = 0.001
    layer_hidden_sizes = [64, 32]
    inference_layer_sizes = [64, 16]
    encoder_timestep = 24
    decoder_timestep = 6
    input_keep_prob = 0.75
    output_keep_prob = 0.75
    fc_keep_prob = 0.75



is_train = True

model_path = 'model/uber_model.ckpt'

# ----------------- Dataset process ----------
records = pd.read_csv('data/charge.csv')

scaler = MinMaxScaler()
records['power'] = scaler.fit_transform(records['power'])

train_records = records.iloc[24 * 30 :]
test_records = records.iloc[  : 24 * 30 ]


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


class UberLSTM():
    def __init__(self, config, scope = 'Uber'):
        # basic assignment
        self.config = config
        last_hidden_size = config.layer_hidden_sizes[-1]

        # general placeholder
        self.input_keep_prob = tf.placeholder(tf.float32)
        self.output_keep_prob = tf.placeholder(tf.float32)
        self.fc_keep_prob = tf.placeholder(tf.float32)
        # encoder model
        with tf.variable_scope(scope + '_' + 'Encoder'):
            encoder_X = tf.placeholder(tf.float32, shape = [None, config.encoder_timestep, input_size])
            
            encoder_lstms = [tf.contrib.rnn.BasicLSTMCell(hidden_size) for hidden_size in config.layer_hidden_sizes]
            encoder_dropout = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = self.input_keep_prob, output_keep_prob = self.output_keep_prob) for lstm in encoder_lstms]
            encoder_cells = tf.contrib.rnn.MultiRNNCell(encoder_dropout, state_is_tuple = True)

            encoder_initial_state = encoder_cells.zero_state(batch_size, tf.float32)
            encoder_all_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cells, encoder_X, initial_state = encoder_initial_state)

            encode_outputs = encoder_all_outputs[:, -1, :]

        # decoder model
        with tf.variable_scope(scope + '_' + 'Decoder'):
            decoder_X = tf.placeholder(tf.float32, shape = [None, config.decoder_timestep, input_size])
            decoder_Y = tf.placeholder(tf.float32, shape = [None, config.decoder_timestep])

            decoder_lstms = [tf.contrib.rnn.BasicLSTMCell(hidden_size) for hidden_size in config.layer_hidden_sizes]
            decoder_dropout = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = self.input_keep_prob, output_keep_prob = self.output_keep_prob) for lstm in decoder_lstms]
            decoder_cells = tf.contrib.rnn.MultiRNNCell(decoder_dropout, state_is_tuple = True)

            decoder_initial_state = encoder_final_state
            decoder_all_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cells, decoder_X, initial_state = decoder_initial_state)

            # full connect to compute x
            decoder_out_W = tf.Variable(tf.truncated_normal(shape = [last_hidden_size * config.decoder_timestep, config.decoder_timestep]))
            decoder_out_b = tf.Variable(tf.constant(0.0, shape = [config.decoder_timestep], dtype = tf.float32))

            decoder_outputs_flat = tf.reshape(decoder_all_outputs, [-1, last_hidden_size * config.decoder_timestep])

            decoder_losses = tf.matmul(decoder_outputs_flat, decoder_out_W) + decoder_out_b

            decoder_losses = tf.losses.mean_squared_error(decoder_losses, decoder_Y)

            decoder_train_op = tf.train.AdamOptimizer(config.lr).minimize(decoder_losses)            

        # inference model with dropout
        with tf.variable_scope('scope' + '_' + 'Inference'):
            Y = tf.placeholder(tf.float32, shape = [None, output_size])

            # get input of inference model
            inference_inputs = tf.reshape(encode_outputs, [-1, last_hidden_size * output_size])

            # build multi-layer full connect nn
            last_size = last_hidden_size
            y_pre = None
            for size in config.layer_hidden_sizes:
                inference_fc_w = tf.Variable(xavier_init(last_size, size))
                inference_fc_b = tf.Variable(tf.constant(0.0, shape = [size], dtype = tf.float32))

                # compute for next loop
                last_size = size
                if y_pre is None:
                    y_pre = tf.nn.dropout(tf.nn.tanh( tf.matmul(inference_inputs, inference_fc_w) + inference_fc_b ), self.fc_keep_prob)
                else:
                    y_pre = tf.nn.dropout(tf.nn.tanh( tf.matmul(y_pre, inference_fc_w) + inference_fc_b ), self.fc_keep_prob)
                
            # comput output
            inference_fc_w = tf.Variable(xavier_init(last_size, output_size))
            inference_fc_b = tf.Variable(tf.constant(0.0, shape = [output_size], dtype = tf.float32))
            inference_y_pre = tf.matmul(y_pre, inference_fc_w) + inference_fc_b

            inference_losses = tf.losses.mean_squared_error(inference_y_pre, Y)
            inference_train_op = tf.train.AdamOptimizer(config.lr).minimize(inference_losses)

        # assignment
        self.encoder_X = encoder_X
        self.decoder_X = decoder_X
        self.decoder_Y = decoder_Y
        self.Y = Y
        
        self.decoder_losses = decoder_losses
        self.decoder_train_op = decoder_train_op
        
        self.inference_losses = inference_losses
        self.inference_train_op = inference_train_op
        
        self.inference_y_pre = inference_y_pre

    def train_ae(self, records, epochs, sess):
        # train ae
        for i in range(epochs):
            ae_batch_data = get_ae_batch_data(records, batch_size, self.config.encoder_timestep, self.config.decoder_timestep)
            all_loss = []
            for batch_data in ae_batch_data:
                loss, _ = sess.run([self.decoder_losses, self.decoder_train_op], feed_dict = {self.encoder_X : batch_data[0], self.decoder_X : batch_data[1], self.decoder_Y : batch_data[2],
                                                                                              self.input_keep_prob : self.config.input_keep_prob,
                                                                                              self.output_keep_prob : self.config.output_keep_prob,
                                                                                              self.fc_keep_prob = self.config.fc_keep_prob})
                all_loss.append(loss)

            print('Encode-Decode train Epoch', i, 'Loss:', np.mean(all_loss))

    def train_inference(self, records, epochs, sess):
        # train inference
        for i in range(epochs):
            series_batch_data = get_series_time_batch_data(records, batch_size, self.config.encoder_timestep)
            all_loss = []
            for batch_data in series_batch_data:
                loss, _ = sess.run([self.inference_losses, self.inference_train_op], feed_dict = {self.encoder_X : batch_data[0], self.Y : batch_data[1],
                                                                                                  self.input_keep_prob : self.config.input_keep_prob,
                                                                                                  self.output_keep_prob : self.config.output_keep_prob,
                                                                                                  self.fc_keep_prob = self.config.fc_keep_prob})
                all_loss.append(loss)

            print('Inference train epoch', i, 'Loss:', np.mean(all_loss))

    def predict(self, records, sess, display = False):
        # test
        test_series_batch_data = get_series_time_batch_data(records, batch_size, self.config.encoder_timestep, shuffle = False)

        y_list = []
        y_pre_list = []
        all_loss = []
        for batch_data in test_series_batch_data:
            loss, pred = sess.run([self.inference_losses, self.inference_y_pre], feed_dict = {self.encoder_X : batch_data[0], self.Y : batch_data[1],
                                                                                              self.input_keep_prob : 1.0,
                                                                                              self.output_keep_prob : 1.0})
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
            
        

# main process

# modle generate

series_model = UberLSTM(SeriesConfig)

sess = sess = tf.Session()
saver = tf.train.Saver()

if is_train:
    sess.run(tf.global_variables_initializer())
    series_model.train_ae(train_records, 300, sess)
    series_model.train_inference(train_records, 500, sess)
             
else:
    saver.restore(sess, model_path)
    

