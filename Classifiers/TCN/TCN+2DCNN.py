
extract_path = '../../NewData/AllData/4.proj/Train_'
checkpoint_model_path= "../../models/"
model_fn = 'proj_tcn_cnn.h5'
MAX_EPOCHS = 500

sub_dirs=['Circle', 'Spread', 'Spin', 'ForwardBack', 'downToUp', 'upToDown', 'leftToRight', 'rightToLeft', 'diag-LeftToRight', 'diag-RightToLeft']

import glob
import os
import numpy as np
# random seed.
rand_seed = 1
from numpy.random import seed
seed(rand_seed)
import tensorflow
tensorflow.random.set_seed(rand_seed)

import keras
from tcn import TCN, tcn_full_summary
from keras.models import Sequential

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Conv1D, MaxPooling2D, Dense, Dropout, Flatten, Bidirectional, TimeDistributed
from sklearn.model_selection import train_test_split
from keras.models import load_model

import gc

# if time_steps > tcn_layer.receptive_field, then we should not
# be able to solve this task.
img_size = 32
time_steps, input_dim = 27, img_size*img_size*1


def one_hot_encoding(y_data, sub_dirs):
    Mapping=dict()

    count=0
    for i in sub_dirs:
        Mapping[i]=count
        count=count+1

    y_features2=[]
    for i in range(len(y_data)):
        Type=y_data[i]
        lab=Mapping[Type]
        y_features2.append(lab)

    y_features=np.array(y_features2)
    y_features=y_features.reshape(y_features.shape[0],1)
    from tensorflow.keras.utils import to_categorical
    y_features = to_categorical(y_features)

    return y_features



def full_model(input_x, input_y):
    print('building the model ... ')
    model = Sequential()
    model.add(TimeDistributed((Conv2D(32, (3, 3), strides=(1, 1), input_shape=(img_size, img_size,1), padding="same", activation="relu", name="Conv2D_1"))))
    model.add(TimeDistributed(MaxPooling2D(strides=(3, 3), pool_size=(3, 3), padding="valid", name="MaxPool_1")))

    model.add(TimeDistributed((Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu", name="Conv2D_2"))))
    model.add(TimeDistributed(MaxPooling2D(strides=(3, 3), pool_size=(3, 3), padding="valid", name="MaxPool_2")))

    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.1))
    tcn_input_layer = Conv1D(input_shape=(time_steps, input_dim), filters=32, kernel_size=1, padding='causal', activation=None)
    tcn_layer = TCN(nb_filters=64, kernel_size=3, nb_stacks=1, dilations=(1, 2, 4),
                    padding='causal', use_skip_connections=True, activation='relu', dropout_rate=0.1)

    # The receptive field tells you how far the model can see in terms of timesteps.
    print('Receptive field size =', tcn_layer.receptive_field)
    model.add(tcn_input_layer)
    model.add(tcn_layer)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(input_y.shape[1], activation='softmax', name = 'output'))

    return model

frame_tog = [27]

# ================================================================================================================
#loading the train data
def read_datasetFile(labelname):
    Data_path = extract_path+labelname
    data = np.load(Data_path+'.npz')
    # float64를 float32로 줄임 : 메모리 최적화를 위해....
    return np.array(data['arr_0'], dtype=np.dtype(np.float32)), data['arr_1']


def dataset_load(): # 학습 데이터 파일에서 읽자!~ 지역변수 사용하여 함수 호출 종료 후 메모리 자동삭제 될 수 있도록 함수로 구현
    train_data, train_label = read_datasetFile(sub_dirs[0])
    print('1= ', train_data.shape,'accumulate : ',train_label.shape)

    for idx, sub_dir in enumerate(sub_dirs[1:]):
        temp_train_data, temp_train_label = read_datasetFile(sub_dir)
        train_data = np.concatenate((train_data, temp_train_data), axis=0)
        train_label = np.concatenate((train_label, temp_train_label), axis=0)
        print(str(idx+2)+'= ', temp_train_data.shape, 'accumulate : ', train_label.shape)
    train_label = one_hot_encoding(train_label, sub_dirs)
    return train_data,train_label

train_data,train_label  = dataset_load()

train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], train_data.shape[3], 1)

print('Training Data Shape is:')
print(train_data.shape,train_label.shape)

X_train, X_val, y_train, y_val  = train_test_split(train_data, train_label, test_size=0.20, random_state=1)
del train_data,train_label


# ==============================================================================================================================

model = full_model(X_train,y_train)

print("Model building is completed")


adam = tensorflow.keras.optimizers.Adam()

model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=adam,
                  metrics=['accuracy'])

tcn_full_summary(model, expand_residual_blocks=False)

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=40, restore_best_weights=True)
# Training the model
learning_hist = model.fit(X_train, y_train,
                             batch_size=32,
                             epochs=MAX_EPOCHS,
                             verbose=1,
                             shuffle=True,
                             validation_data=(X_val,y_val),
                             callbacks=[callback]
                          )
model.summary()
model.save(model_fn)

"""
import matplotlib.pyplot as plt
# 6 훈련 과정 시각화 (정확도)
plt.plot(learning_hist.history['accuracy'])
plt.plot(learning_hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# 7 훈련 과정 시각화 (손실)
plt.plot(learning_hist.history['loss'])
plt.plot(learning_hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
"""