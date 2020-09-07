preprocess_path = "./Data-Science-template-master/Code/preprocessing"
import sys
import os
#print(os.listdir())
sys.path.append(preprocess_path)
from preprocess import import_Xy
#import joblib
import logging
import tensorflow as tf
import numpy as np
import joblib

def cnv_to_OneHot(data, num_of_classes):        # Input - Numpy array with 1 dimension or (i,) dimension

    x = data.shape
    data_one_hot = np.zeros((x[0], num_of_classes))
    for i in range(x[0]):
        data_one_hot[i, data[i]] = 1
    print(type(data_one_hot[0,0]))
    return data_one_hot


def train_model():
    X_train,y_train,X_test,y_test = import_Xy()
    #from sklearn.svm import SVC
    #model = SVC(kernel='linear')
    model = tf.keras.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu', input_shape=(28,28,1)),
                                    tf.keras.layers.MaxPool2D(strides = 2),
                                    tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(strides = 2),
                                    #tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu'),
                                    #tf.keras.layers.MaxPool2D(strides = 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation = 'relu'),
                                    #tf.keras.layers.Dense(64, activation = 'relu'),
                                    tf.keras.layers.Dense(32, activation = 'relu'),
                                    tf.keras.layers.Dense(10, activation = 'relu')
        
                                   ])
    model.load_weights("task_1_saved")
    model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(X_train, y_train, epochs = 1, validation_data = (X_test, y_test))
    #model.save_weights("task_1_saved")
#   y_test_OneHot = cnv_to_OneHot(y_test, 10)
    #print(y_test[5])
    #print(y_test_OneHot[5])
    from sklearn.metrics import precision_score
    y_pred = model.predict(X_test)
    y_pred = tf.math.argmax(y_pred, 1)
    precision = precision_score(y_test, y_pred,average='weighted')

    return model,precision 

def save_model(model,precision,model_path = "Data-Science-template-master/Data/models/latest_model.h5"):
    
    print("Precision: {}".format(precision))
    model.save(model_path)
    #joblib.dump(model,model_path)
    #del model

if __name__ == "__main__":
    model, precision = train_model()
    #print(os.listdir(""))
    save_model(model, precision)


#train_model()
 