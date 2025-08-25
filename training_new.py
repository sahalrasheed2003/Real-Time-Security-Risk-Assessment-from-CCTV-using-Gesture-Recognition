from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model

# from keras.preprocessing.image import ImageDataGenerator

import numpy as np

#------------------------------
# sess = tf.Session()
# keras.backend.set_session(sess)
#------------------------------
#variables
num_classes =2
batch_size = 100
epochs = 12
#------------------------------

# from tensorflow import keras
import os, cv2
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
# from tensorflow.keras.engine.saving import load_model
# manipulate with numpy,load with panda
import numpy as np

# data visualization
import cv2


def read_dataset():
    data_list = []
    label_list = []
    i=0
    my_list = os.listdir(r'D:\project\Risk_assessment\myapp\static\dataset')
    for pa in my_list:  #   iterate inside folder
        print(pa,"==================",i)
        for root, dirs, files in os.walk(r'D:\project\Risk_assessment\myapp\static\dataset\\' + pa):
            for f in files:
                 print(pa, "------------>", f)
                 vs = cv2.VideoCapture(r'D:\project\Risk_assessment\myapp\static\dataset\\'+pa+"\\" + f)
                 cnt=0
                 while True:
                     cnt=cnt+1
                     ok, frame = vs.read()
                     if ok:
                         cv2.imwrite(r'D:\project\Risk_assessment\myapp\static\cap.jpg', frame)
                         img = cv2.imread(r'D:\project\Risk_assessment\myapp\static\cap.jpg', cv2.IMREAD_GRAYSCALE)
                         res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
                         data_list.append(res)
                         label = i
                         label_list.append(label)
                     if cnt>=50:
                         break
        i = i + 1
    return (np.asarray(data_list, dtype=np.float32), np.asarray(label_list))




model = Sequential()

# 1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

# 3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())

# fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
# ------------------------------
# batch process


import tensorflow as tf
# import keras
from sklearn.model_selection import train_test_split
# load dataset
x_dataset, y_dataset = read_dataset()
X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)

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


print(x_train.shape)

gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

# ------------------------------

model.compile(loss='categorical_crossentropy'
              , optimizer="adam"
              , metrics=['accuracy']
              )

# ------------------------------

if not os.path.exists(r"D:\project\Risk_assessment\myapp\static\fight_model.h5"):

    model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)
    model.save(r"D:\project\Risk_assessment\myapp\static\fight_model.h5")  # train for randomly selected one
else:
    model = load_model(r"D:\project\Risk_assessment\myapp\static\fight_model.h5")  # load weightsh5")  # load weights
