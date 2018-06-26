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
autoencoder_input_size = 12

series_timesteps = 6
series_hidden_size = 64
series_layer_num = 1

output_size = 1

batch_size = 24

keep_prob = 0.75

is_train = False

model_path = 'model/model.ckpt'
ae_model_path = 'model/ae_model.ckpt'

# ----------------- Dataset process ----------
records = pd.read_csv('data/charge.csv')

scaler = MinMaxScaler()
records['power'] = scaler.fit_transform(records['power'])

train_records = records.iloc[: -24 * 5]
test_records = records.iloc[:]


def xavier_init(fat_in, fat_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fat_in + fat_out) )
    high = constant * np.sqrt(6.0 / (fat_in + fat_out) )
    return tf.random_uniform( (fat_in, fat_out), minval = low, maxval = high, dtype = tf.float32)

class lstm_model():
    def __init__(self, input_size, output_size, hidden_size, layer_num, timesteps, batch_size, keep_prob, name_scope = 'lstm'):
        with tf.name_scope(name_scope):
            # build model
            X_ = tf.placeholder(tf.float32, shape = [None, timesteps, input_size])
            
            
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
            
    def train_autoencorder(sess):
 
        target_size = series_timesteps
        en_w = tf.Variable(tf.truncated_normal([target_size, 24], stddev = 0.01), dtype = tf.float32)
        en_b = tf.Variable(tf.constant(0, shape = [24], dtype = tf.float32))

        de_w = tf.Variable(tf.truncated_normal([24, target_size], stddev = 0.01), dtype = tf.float32)
        de_b = tf.Variable(tf.constant(0, shape = [target_size], dtype = tf.float32))

        X_INPUT = tf.placeholder(tf.float32, shape = [None, series_timesteps, input_size])

        input_origin = tf.reshape(X_INPUT, [-1, series_timesteps])
        input_en = tf.matmul(input_origin, en_w) + en_b

        output_de = tf.matmul(input_en, de_w) + de_b

        loss = tf.losses.mean_squared_error(input_origin, output_de)

        train_option = tf.train.AdamOptimizer().minimize(loss)


        for i in range(120):
            batchs_data = get_series_time_batch_data(train_records, batch_size, series_timesteps)
            for batch in batchs_data:
                # train
                sess.run([train_option], feed_dict = {X_INPUT : batch[0]})

    
class AdditiveGaussianNoisAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        # parameters assignment
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # build model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add( tf.matmul(self.x + self.scale * tf.random_normal((n_input,)), self.weights['w1']),
            self.weights['b1']))

        self.reconstruction = tf.add( tf.matmul( self.hidden, self.weights['w2']), self.weights['b2'] )

        # optimize
        self.cost = 0.5 * tf.reduce_sum( tf.pow(tf.subtract(self.reconstruction, self.x), 2.0) )
        self.optimizer = optimizer.minimize(self.cost)

        # initialize session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))

        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run( (self.cost, self.optimizer), feed_dict = {self.x : X, self.scale : self.training_scale})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run( self.cost, feed_dict = {self.x : X, self.scale : self.training_scale})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x : X, self.scale : self.training_scale})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.recontruction, feed_dict = {self.hidden : hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, fedd_dict = {self.x : X, self.scale : self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiasses(self):
        return self.sess.run(self.weights['b1'])


class LstmAutoEncoder(object):
    def __init(self, input_size, encode_timesteps, decode_timesteps, hidden_size)

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

# ---------- Model build --------

y = tf.placeholder(tf.float32, [None, output_size])

# build lstm layers
series_model = lstm_model(input_size = input_size, output_size = output_size, hidden_size = series_hidden_size, layer_num = series_layer_num, timesteps = series_timesteps, batch_size = batch_size, keep_prob = keep_prob, name_scope='series')

output_W = tf.Variable(tf.truncated_normal([1, 1], stddev = 0.01), dtype = tf.float32)
output_b = tf.Variable(tf.constant(0.1, dtype = tf.float32))


y_pre = tf.matmul(series_model.y_pre, output_W) + output_b

loss_func = tf.losses.mean_squared_error(y, y_pre)

train_op = tf.train.AdamOptimizer().minimize(loss_func)


saver = tf.train.Saver()
def auto_encoder():

    ae = AdditiveGaussianNoisAutoencoder(autoencoder_input_size, series_timesteps)
    for i in range(120):
        batchs_data = get_series_time_batch_data(train_records, batch_size, autoencoder_input_size)
        all_loss = []
        for batch in batchs_data:
            features = batch[0].reshape((-1, autoencoder_input_size))
            ls = ae.partial_fit(features)
            all_loss.append(ls)

        print('AE epoch', i, 'Loss:', np.mean(all_loss))

    saver.save(ae.sess, ae_model_path)
    return ae

if is_train:
    ae = auto_encoder()
else:
    ae = AdditiveGaussianNoisAutoencoder(autoencoder_input_size, series_timesteps)
    saver.restore(ae.sess, ae_model_path)

def predict():
    
    with tf.Session() as sess:
        # restore model
        saver.restore(sess, model_path)
        test_data = get_series_time_batch_data(test_records, batch_size, autoencoder_input_size, False)
        
        test_y_list = [] 
        test_y_pre_list = []
        test_all_loss = []
        for batch in test_data:
            # auto encoding
            features = batch[0].reshape((-1, autoencoder_input_size))
            features = ae.transform(features)
            features = features.reshape((-1, series_timesteps, input_size))
            
            predict, loss = sess.run([y_pre, loss_func], feed_dict = {series_model.X_ : features, y : batch[1]})
            test_y_list.extend(batch[1])
            test_y_pre_list.extend(predict)
            test_all_loss.append(loss)

        # display
        test_x = list(range(len(test_y_list)))

        # inverse normalization
        test_y_list = np.array(scaler.inverse_transform(test_y_list))
        test_y_pre_list = np.array(scaler.inverse_transform(test_y_pre_list))

        mse = np.mean( (test_y_list - test_y_pre_list) ** 2)
        print('---------------- Test Loss:', np.mean(test_all_loss), 'RMSE:', np.sqrt( mse) )
        plt.plot(test_x, test_y_list, 'r', test_x, test_y_pre_list, 'b')
        plt.show()
        

def train():
    # ----------------- Train ----------

    with tf.Session() as sess:
        print('Count of parameters:', np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()]))
        min_loss = 100000
        saver = tf.train.Saver()
        for i in range(10000):
            y_list = [] 
            y_pre_list = []
            all_loss = []
            sess.run(tf.global_variables_initializer())
            
            batchs_data = get_series_time_batch_data(train_records, batch_size, autoencoder_input_size)
            for batch in batchs_data:
                # auto encoding
                features = batch[0].reshape((-1, autoencoder_input_size))
                features = ae.transform(features)
                features = features.reshape((-1, series_timesteps, input_size))
                # train
                sess.run([train_op], feed_dict = {series_model.X_ : features, y : batch[1]})
                predict, loss = sess.run([y_pre, loss_func], feed_dict = {series_model.X_ : features, y : batch[1]})
                

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

if is_train:
    train()
else:
    predict()
