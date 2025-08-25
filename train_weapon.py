from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model

#variables
num_classes =7
batch_size = 15
epochs = 20
#------------------------------
import os, cv2
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
# from tensorflow.keras.engine.saving import load_model
# manipulate with numpy,load with panda
import numpy as np
# import pandas as pd

# data visualization
import cv2
# get_ipython().run_line_magic('matplotlib', 'inline')

proj_path=r"D:\project\Risk_assessment\myapp\static\\"
# Data Import
def read_dataset():
    data_list = []
    label_list = []
    i=0
    my_list = os.listdir(proj_path + 'dataset_weapon_new')
    for pa in my_list:
        print(pa,"==================",i)
        for root, dirs, files in os.walk(proj_path + 'dataset_weapon_new\\' + pa):

         for f in files:
            file_path = os.path.join(proj_path + 'dataset_weapon_new\\'+pa, f)

            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
            data_list.append(res)
            label = i
            label_list.append(label)
        i=i+1
    return (np.asarray(data_list, dtype=np.float32), np.asarray(label_list))

def read_dataset1(path):
    data_list = []
    label_list = []

    file_path = os.path.join(path)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    data_list.append(res)
    # label = dirPath.split('/')[-1]

            # label_list.remove("./training")
    return (np.asarray(data_list, dtype=np.float32))

def train_weapon():
    from sklearn.model_selection import train_test_split
    # load dataset
    x_dataset, y_dataset = read_dataset()
    X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)

    import tensorflow as tf
    y_train1=[]
    for i in y_train:
        emotion = tf.keras.utils.to_categorical(i, num_classes)
        print(i,emotion)
        y_train1.append(emotion)

    y_train=y_train1
    x_train = np.array(X_train, 'float32')
    y_train = np.array(y_train, 'float32')
    x_test = np.array(X_test, 'float32')
    y_test = np.array(y_test, 'float32')

    x_train /= 255  # normalize inputs between [0, 1]
    x_test /= 255
    print("x_train.shape",x_train.shape)
    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype('float32')

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # ------------------------------
    # construct CNN structure

    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    # ------------------------------
    # batch process

    gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

    # ------------------------------

    model.compile(loss='categorical_crossentropy'
                  , optimizer="adam"
                  , metrics=['accuracy']
                  )

    # ------------------------------

    if not os.path.exists(proj_path + "model_weapon_new2.h5"):
        model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)
        model.save(proj_path + "model_weapon_new2.h5")  # train for randomly selected one
    else:
        model = load_model(proj_path + "model_weapon_new2.h5")  # load weights

    print("Training completed")
# train_weapon()

def predict_cnn_weapon(path):
    dataset = read_dataset1(path)
    dataset = dataset / 255
    (mnist_row, mnist_col, mnist_color) = 48, 48, 1
    dataset = dataset.reshape(dataset.shape[0], mnist_row, mnist_col, mnist_color)
    mo = load_model(proj_path + "model_weapon_new2.h5")
    yhat_classes = mo.predict_classes(dataset, verbose=0)
    yhat_score = mo.predict(dataset, verbose=0)
    # if yhat_classes[0] == 6 or yhat_classes[0] == 3:
    #     return "No", yhat_score[0][yhat_classes[0]]
    # else:
    return yhat_classes[0], yhat_score[0][yhat_classes[0]]
