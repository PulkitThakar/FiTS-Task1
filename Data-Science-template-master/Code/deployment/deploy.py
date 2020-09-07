from flask import Flask, jsonify
from flask import abort
from flask import request
import numpy as np
import json
import os
import joblib
import tensorflow as tf
#MODEL_PATH = "../../Data/models/latest_model.pkl"
MODEL_PATH = "Data-Science-template-master/Data/models/latest_model.h5"


clf = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

print("launching the server....")

@app.route('/api', methods=["POST"])
def ModelEval(verbose = True):
    if request.is_json:

        # Parse the JSON into a Python dictionary of the right form
        req = request.get_json()
        X_eval = np.array(req["data"]).reshape((1,28,28,1))
        # Print the dictionary
        
        prediction = clf.predict(X_eval)
        #prediction = tf.math.argmax(prediction, 1)
        print("prediction", prediction)
        print(type(prediction))
        prediction = np.argmax(prediction, axis = 1)
        print(type(prediction))
        print(prediction)
        # Return a string along with an HTTP status code
        print(type(prediction.tolist()[0]))
        return jsonify(prediction.tolist()), 200

    else:

        # The request body wasn't JSON so return a 400 HTTP status code
        return "Request was not JSON", 400
  



if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '443'))
    except ValueError:
        PORT = 443
    app.run()
