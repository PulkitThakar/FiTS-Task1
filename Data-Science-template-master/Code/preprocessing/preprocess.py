import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def import_Xy(path_train="./Data-Science-template-master/Data/fashion-mnist_train.csv",path_test="./Data-Science-template-master/Data/fashion-mnist_test.csv",label_name="label"):
    # importng the data from the paths which are there by default
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    X_train,y_train = df_train.drop("label",axis=1).values,df_train["label"].values
    X_train = np.reshape(X_train, (60000, 28, 28,1))
    X_test,y_test = df_test.drop("label",axis=1).values,df_test["label"].values
    X_test = np.reshape(X_test, (10000, 28, 28,1))
    #plt.imshow(X_test[0,:,:,0], cmap = 'binary')
    #plt.show()
    return X_train,y_train, X_test,y_test

if __name__=="__main__":
    X_train, y_train, X_test, y_test = import_Xy()
    np.savetxt("test_sample.txt"  , X_test[0,:,:,0])
    print("imported data")

#if __name__=="__main__":
#    import_Xy()
#    print("imported data")