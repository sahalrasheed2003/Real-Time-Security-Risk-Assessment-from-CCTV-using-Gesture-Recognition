
num_classes =2
batch_size = 40
epochs = 2
#------------------------------



import os
from tensorflow.python.keras.models import load_model
# manipulate with numpy,load with panda
import numpy as np
# import pandas as pd

# data visualization
import cv2

# get_ipython().run_line_magic('matplotlib', 'inline')

def read_dataset1(path):
    data_list = []
    label_list = []

    file_path = os.path.join(path)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    data_list.append(res)
    return (np.asarray(data_list, dtype=np.float32))

# from keras import backend as K
def predictcnn(fn):
    dataset=read_dataset1(fn)
    (mnist_row, mnist_col, mnist_color) = 48, 48, 1

    dataset = dataset.reshape(dataset.shape[0], mnist_row, mnist_col, mnist_color)
    mo = load_model(r"D:\project\Risk_assessment\myapp\static\fight_model.h5")
    dataset /= 255
    # predict probabilities for test set

    yhat_classes = mo.predict_classes(dataset, verbose=0)
    result=yhat_classes[0]
    res=int(yhat_classes.tolist()[0])        #numpy ndarray type convert into list type
    # K.clear_session()                        #session clear(avoid simultaneously running error)
    return result
    # return result





#
#     print(yhat_classes)
#
# predictcnn(r"D:\project\Risk_assessment\myapp\static\20240318_144250_2.png")
