# list of the check functions, which have to output a boolean, with True if the  test passes
import sys
code_path = "Data-Science-template-master/Code"
sys.path.append("{}/preprocessing".format(code_path))
sys.path.append("{}/training".format(code_path))
model_save_path = "Data-Science-template-master/Data/models"
#sys.path.append(model_save_path)
saved_file_name = "latest_model.h5"
import preprocess
import train
import requests
import json
import numpy as np
import os
import tensorflow as tf

checkdeploymentError=""


def checkdeployment():
    # function which tests whether the deployment is successful
    try :
        #Creating a sample data for fashion MNIST (28*28 images)
        a, b, c, d = preprocess.import_Xy()
        #print(c.shape)
        sample_data = np.empty((1,28,28,1))
        sample_data[0] = c[0,:,:,:]
        del a, b, c, d
        print(sample_data.shape)
        sample_data = sample_data.tolist()
        #sample_data = [0 for k in range(784)]

        #headers = {'Content-Type': 'application/json'}
        r = requests.post(url = "http://127.0.0.1:5000/api", json= {"data": sample_data})#, headers = headers)
        r.json()
        result = (type(r.json()[0])==int)
    except :
        result = False
    return result

def checkTrainingDataFormat():

    try:
        X_train,y_train, X_test,y_test = preprocess.import_Xy()

        result = len(X_train.shape)==4 and X_train.shape[1]==28 and X_train.shape[2]==28 and X_train.shape[3]==1
    except:
        result = False
    return result

def checkModelSaving():

    try:
        x,y = train.train_model()
        train.save_model(x,y, "{}/{}".format(model_save_path, "test.h5"))
        result = os.path.exists("{}/{}".format(model_save_path, "test.h5"))
        os.remove("{}/{}".format(model_save_path, "test.h5"))
    except:
        result = False
    return result

def checkPrecision():
    try:
        x,y = train.train_model()
        y = y.item()
        result = type(y)==float and y>0.87
    except:
        result = False
    return result

def checkTrainingMethod():
    try:
        model, precision = train.train_model()
        result = (type(model) == tf.python.keras.engine.sequential.Sequential and type(precision.item()) == float)
    except:
        result = False
    return result
    