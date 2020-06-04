from flask import Flask, request, Response, jsonify
from keras.models import model_from_json
import pandas as pd
import json
import numpy as np
import ast
import flask_json
from scipy.stats import entropy
import math
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


global loaded_model_accel_lstm
global loaded_model_gyro_lstm
global loaded_model_magn_lstm


lables = ['texting + game', 'rest', 'pocket', 'running', 'shaking', 'walking']

########### add model path
path = 'Models_lstm'
app = Flask(__name__)


@app.before_request
def load_model_func():
    global loaded_model_accel_lstm
    global loaded_model_gyro_lstm
    global loaded_model_magn_lstm
    loaded_model_accel_lstm = load_model('/model_without_brand_21.json', '/model_without_brand_21.h5')
    # loaded_model_gyro_lstm = load_model('/model_without_brand.json', '/model_without_brand.h5')
    # loaded_model_magn_lstm = load_model('/model_without_brand.json', '/model_without_brand.h5')



def load_model(model_name, weights_name):
    path_model = path + model_name
    path_weights = path + weights_name
    json_file = open(path_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path_weights)
    print("Loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


@app.route('/', methods=['GET', 'POST'])
def main():
    return 'SDI action classifier is in the House!'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        global loaded_model_accel_lstm
        data_b = request.get_data()
        data_s = str(data_b, 'utf-8')
        data_s = data_s.replace("\r", "")
        data_s = data_s.replace("\n", "")
        data_s = data_s.replace('\\', "")
        if(data_s[0]=='"'):
            data_s = data_s[:0] + data_s[1:]
        if (data_s[len(data_s)-1] == '"'):
            data_s = data_s[:len(data_s)-1]
        data = json.loads(data_s)
        accel = data['accel']
        #gyro = data['gyro']
        #magn = data['magn']

        df_accel = get_data_as_df(accel)
        #df_gyro = get_data_as_df(gyro)
        #df_magn = get_data_as_df(magn)

        ms = '500ms'

        df_accel = resample_df(df_accel, ms)
        #df_gyro = resample_df(df_gyro, ms)
        #df_magn = resample_df(df_magn, ms)

        df_accel = reset_index(df_accel)
        #df_gyro = reset_index(df_gyro)
        #df_magn = reset_index(df_magn)

        print(len(df_accel.columns))
        df_accel = create_dataset(df_accel)
        #df_gyro = create_dataset(df_gyro)
        #df_magn = create_dataset(df_magn)

        prob = loaded_model_accel_lstm.predict(df_accel)

        pred = np.argmax(prob, axis=-1)

        print(pred[0])
        resp = Response(lables[pred[0]], status=200, mimetype='application/json')
    except Exception as e:
        resp = Response(e, status=500, mimetype='application/json')
    return resp

def reset_index(df):
    index = df.index
    df = df.reset_index(drop=True)
    df['t'] = index
    df = df.replace([np.inf, -np.inf], -1)
    return df

def create_dataset(ds, window_size=50, steps=10, features=range(0,21)):
    X = []

    group = ds
    group = group.drop(['t'], axis=1)
    group = np.array(group)
    pointer = 0
    print(group.shape)
    while pointer < (group.shape[0]-window_size):
        window_x = group[pointer:pointer+window_size, features]
        window_x = window_x.reshape((len(window_x)),len(features))
        X.append(window_x)
        pointer += steps
    print(np.array(X).shape)
    return np.array(X)

def get_data_as_df(array_to_load):
    df = pd.DataFrame(array_to_load)
    return df


def resample_df(df, ms):
    print(df.shape)
    five_hz_mean_accel = pd.DataFrame()
    group = df.set_index(['t'])
    group.index = pd.to_datetime(group.index, unit='ms')
    resample = group.resample(ms)
    five_hz_mean = pd.DataFrame()
    five_hz_mean = resample.mean()
    count = resample.count()
    sum_ = resample.sum()
    delta = sum_[['x', 'y', 'z']] - five_hz_mean[['x', 'y', 'z']]
    abs_delta = abs(delta)
    count = count['x'].to_numpy()
    five_hz_mean['sma'] = np.array(abs_delta.sum(axis=1).to_numpy()/count)
    five_hz_mean['std_x'] = resample['x'].std()
    five_hz_mean['std_y'] = resample['y'].std()
    five_hz_mean['std_z'] = resample['z'].std()
    five_hz_mean['mad_x'] = resample['x'].mad()
    five_hz_mean['mad_y'] = resample['y'].mad()
    five_hz_mean['mad_z'] = resample['z'].mad()
    five_hz_mean['max_x'] = resample['x'].max()
    five_hz_mean['max_y'] = resample['y'].max()
    five_hz_mean['max_z'] = resample['z'].max()
    five_hz_mean['min_x'] = resample['x'].min()
    five_hz_mean['min_y'] = resample['y'].min()
    five_hz_mean['min_z'] = resample['z'].min()
    five_hz_mean['squares_x'] = pow(resample['x'].mean(), 2)
    five_hz_mean['squares_y'] = pow(resample['y'].mean(), 2)
    five_hz_mean['squares_z'] = pow(resample['z'].mean(), 2)
    five_hz_mean['entropy'] = entropy([resample['x'].mean(), resample['y'].mean(), resample['z'].mean()], base=2)
    five_hz_mean['energy'] = (five_hz_mean['squares_x'] + five_hz_mean['squares_y'] + five_hz_mean['squares_z']) / 3
    five_hz_mean = five_hz_mean.fillna(five_hz_mean.mean())
    # five_hz_mean_accel = pd.concat([five_hz_mean_accel, five_hz_mean])
    return five_hz_mean

if __name__ == "__main__":
    app.run()
