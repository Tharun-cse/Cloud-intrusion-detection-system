from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import cv2
from imutils import paths
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras import regularizers
from keras.losses import MeanSquaredError
from keras import datasets, layers, models
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import keras
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#     import seaborn as sns
#     sns.set()

import re
import string
from wordcloud import WordCloud
from collections import Counter

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import precision_recall_fscore_support

# app = Flask(__name__)
# @app.route('/test')

from flask import Flask, render_template, request

app = Flask(__name__)

# define global variable for file path
file_path = ''

# define function to run prediction
def runi(dataset):
    # your prediction code here
    dff = pd.read_csv(dataset)
    dff = dff.fillna(0)
    dff = dff.replace([np.inf, -np.inf], 1e9)

    # col_to_encode = ' Label'
    # encoder = LabelEncoder()
    # dff[col_to_encode] = encoder.fit_transform(dff[col_to_encode])

  # Separate the features and target
    features = dff.iloc[:, :-1]
    target = dff.iloc[:, -1]

  # Create a MinMaxScaler object
    scaler = MinMaxScaler()

  # Normalize the data in each column
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

  # split into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    #
    # X_train = X_train.values
    # X_test = X_test.values
    # X_train = pd.DataFrame(X_train)
    # X_test = pd.DataFrame(X_test)
    #
    #
    # y_train = np.squeeze(y_train)
    # y_test = np.squeeze(y_test)

    features = features.values
    features = pd.DataFrame(features)
    target = np.squeeze(target)

    # Define the hyperparameters
    input_dim = 78
    hidden_dim_1 = 64
    hidden_dim_2 = 32
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 1
    beta = 1.0  # the coefficient for the contractive penalty term

    # Define the layers of the autoencoder
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoder_1 = tf.keras.layers.Dense(hidden_dim_1, activation="relu")(input_layer)
    encoder_2 = tf.keras.layers.Dense(hidden_dim_2, activation="relu", name='encoder_2')(encoder_1)
    decoder_1 = tf.keras.layers.Dense(hidden_dim_1, activation="relu")(encoder_2)
    decoder_2 = tf.keras.layers.Dense(input_dim, activation="sigmoid")(decoder_1)

    # Define the model and compile it
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder_2)

    def contractive_loss(y_true, y_pred):
        """Calculates the contractive loss for a given batch of input data."""
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        W = K.variable(value=autoencoder.get_layer('dense').get_weights()[0]) # Get the weight matrix of the first hidden layer
        # Compute the jacobian matrix of the hidden layer outputs with respect to the input layer inputs
        h = autoencoder.get_layer('encoder_2').output
        dh = h * (1 - h) # Derivative of the sigmoid activation function
        jacobian = dh[:, None] * W.T[None, :, :] # Compute the jacobian matrix
        jacobian = tf.reduce_sum(tf.square(jacobian), axis=(1, 2))
        return mse + 1e-4 * jacobian


    encoder_1 = tf.keras.models.Model(inputs=input_layer, outputs=encoder_1)
    encoder_2 = tf.keras.models.Model(inputs=encoder_1.input, outputs=encoder_2)
    encoder_2.load_weights('./Models/encoder_2_weights.h5')

    # encoded_train = encoder_2.predict(X_train)
    # encoded_test = encoder_2.predict(X_test)

    encoded_features = encoder_2.predict(features)

    features1 = [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 19, 22, 23, 24, 26, 27, 28, 29, 30, 31]

  # Convert the NumPy array to a pandas DataFrame
    # fea_train = pd.DataFrame(encoded_train)
    # fea_test = pd.DataFrame(encoded_test)
    fea_val = pd.DataFrame(encoded_features)

    # fea_train1= fea_train[features]
    # fea_test1=fea_test[features]
    fea_val1 = fea_val.iloc[:, features1].dropna()

    # fea_train1 = np.array(fea_train1)
    # fea_test1 = np.array(fea_test1)
    fea_val1 = np.array(fea_val1)

    from keras.models import load_model
    model3=load_model('./Models/LSTM_final.h5')
    import joblib
    dt = joblib.load('./Models/dt_model33.joblib')

  # Compute initial ensemble accuracy
    dt_pred1 = dt.predict(fea_val1)

  # accuracy1 = accuracy_score(y_test, dt_pred1)
  # print(accuracy1)
    lstm_pred_prob = model3.predict(fea_val1)
    lstm_pred1 = np.argmax(lstm_pred_prob, axis=1)

    weights=[0.98487615, 0.01706361]

    ensemble_pred1 = np.average([dt_pred1, lstm_pred1], axis=0, weights=weights)
    initial_score = np.mean(ensemble_pred1 == target)

    print("Initial ensemble accuracy: {:.2f}%".format(initial_score*100))

    y_pred_ensemble = np.array([np.argmax(np.bincount([dt_pred1[i], lstm_pred1[i]])) for i in range(len(dt_pred1))])
    accuracy = accuracy_score(y_pred_ensemble, target)
    from collections import Counter
    count = Counter(y_pred_ensemble)
    print(count)
    return "Initial ensemble accuracy: {:.2f}%".format(accuracy*100), count

@app.route('/', methods=['GET', 'POST'])
def index():
    # initialize variables
    global file_path
    output_text = ''
    result = ''
    count={}
    # if user submits form
    if request.method == 'POST':
        # if user clicked upload button
        if 'upload' in request.form:
            # get uploaded file
            uploaded_file = request.files['file']
            # save file to disk
            uploaded_file.save(uploaded_file.filename)
            # set file path variable
            file_path = uploaded_file.filename
            # print message to console
            print(f"File uploaded: {file_path}")
        # if user clicked predict button
        elif 'predict' in request.form:
            if file_path == '':
                output_text = 'Error: No file uploaded'
            else:
                # get prediction result
                result_v, count= runi(file_path)
                output_text = f"Prediction result: {result_v}"
                # print message to console
                result = f"Result Value: {result_v}\n\nResult Dictionary: {count}"
                print(output_text)
            # print message to console
            print(f"Prediction result: {result, count}")
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
