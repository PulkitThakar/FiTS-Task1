import sys
import sys
code_path = "Data-Science-template-master/Code"
sys.path.append("{}/preprocessing".format(code_path))
sys.path.append("{}/training".format(code_path))
import preprocess
import train
import requests
import json
import numpy as np
import os
model_save_path = "Data-Science-template-master/Data/models"

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
print(type(r.json()[0])==int)

print(os.listdir(model_save_path))
result = "latest_model.h5" in os.listdir(model_save_path)
print(result)

saved_file_name = "latest_model.h5"    

print(os.path.exists("{}/{}".format(model_save_path, saved_file_name)))

x,y = train.train_model()
y = y.item()
print(type(y))
result = type(y)==float and y>0.87