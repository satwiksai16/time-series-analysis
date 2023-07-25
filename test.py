import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for
from flask_cors import cross_origin
import pickle
import os
import joblib
import tensorflow as tf


app = Flask(__name__)

df = pd.read_csv('examplers.csv')
df.set_index('date', inplace=True)

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    #start_ind = test_df.head(1).index
    end_ind = df.tail(1).index[0]
    return render_template('model.html', end_date=end_ind)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    n_input = 7
    n_features = 1

    model = tf.keras.models.load_model('model.h5')

    test_df = pd.read_csv('examplers.csv')
    print(test_df)
    test_df.set_index('date', inplace=True)
    # test_df

    scaler_load = joblib.load('scaler.sav')
    scaled_test = scaler_load.transform(test_df)

    test_predictions = []

    first_eval_batch = scaled_test
    current_batch = first_eval_batch.reshape((1, n_input, n_features))

    for i in range(len(test_df)):
        # get the prediction value for the first batch
        current_pred = model.predict(current_batch)[0]

        # append the prediction into the array
        test_predictions.append(current_pred)

        # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    true_predictions = scaler_load.inverse_transform(test_predictions)

    true_predictions.flatten()

    import datetime
    ind = test_df.tail(1).index
    # ind
    date_indices = pd.date_range(ind[0], periods=8)
    exemplers_df = pd.DataFrame({'sales': list(np.round(true_predictions.flatten(), 2))}, index=date_indices[1:])
    exemplers_df.index.name = 'date'
    print(exemplers_df)
    exemplers_df.to_csv('examplers.csv')





    end_ind = test_df.tail(1).index[0]
    pred = list(np.round(true_predictions.flatten()))
    rand = np.random.randint(low=10, high=50, size=len(pred))
    pred = pred + rand
    print('randddddddddd', rand)
    return render_template('model.html', end_date=end_ind, prediction=pred)


if __name__=='__main__':
    app.run()