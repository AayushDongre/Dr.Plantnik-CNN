import cv2
import numpy as np
import os
import pickle
from random import shuffle


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils.np_utils import to_categorical

train_dir = "C:/Users/student/Desktop/Hackathon/train_data/train"
test_dir = "C:/Users/student/Desktop/Hackathon/test_data/test"
image_size = 256
LR = 1e-3

def create_data_set():
    training_data = []
    for i in range(38):
        print("current i="+str(i))
        cur_dir = train_dir + "/c_{}".format(i)
        
        for img in os.listdir(cur_dir):
            label = str(i)
            path = os.path.join(cur_dir,img)
            img = cv2.resize(cv2.imread(path),(image_size,image_size))
            training_data.append([np.array(img),np.array(label)])
    
    shuffle(training_data)
    print(training_data[0])
    
    with open('training_dataB.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    return training_data

print("loading data")
with open('training_dataB.pkl', 'rb') as f:
    training_data = pickle.load(f)
print("data loaded")

train = training_data[:-500]
test = training_data[-500:]

train_x = np.array([i[0] for i in train]).reshape(-1,image_size,image_size,3)
y_train = [i[1] for i in train]
train_y = np.array([i for i in y_train])
train_y1 = to_categorical(train_y)

test_x = np.array([i[0] for i in test]).reshape(-1,image_size,image_size,3)
y_test = [i[1] for i in test]
test_y = np.array([i for i in y_test])
test_y1 = to_categorical(test_y)

print(np.shape(test_x))
print(np.shape(test_y))
print(np.shape(train_x))
print(train_y[0])

model = Sequential()

model.add(Conv3D(32, (3, 3, 3), input_shape=(image_size,image_size,3,1)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2 , 3)))

model.add(Conv3D(32, (3, 3 ,3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2 ,3)))

model.add(Conv3D(64, (3, 3 ,3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2,3)))

model.add(Conv3D(64, (3, 3 ,3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2,3)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.7))

model.add(Dense(38))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(train_x,train_y1,epochs=3,validation_data=(test_x,test_y1))

model.save_weights('model_weights.h5')
model.save('model_keras.h5')
