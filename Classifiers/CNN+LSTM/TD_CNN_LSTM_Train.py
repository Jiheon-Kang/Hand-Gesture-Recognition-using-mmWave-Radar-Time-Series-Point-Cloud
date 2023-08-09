"""
Time-Distributed CNN + Bidirectional LSTMS on VOXELS

- extract_path is the where the extracted data samples are available.
- checkpoint_model_path is the path where to checkpoint the trained models during the training process


EXAMPLE: SPECIFICATION

extract_path = '/Users/sandeep/Research/Ti-mmWave/data/extract/Train_Data_voxels_'
checkpoint_model_path="/Users/sandeep/Research/Ti-mmWave/data/extract/TD_CNN_LSTM"

"""


extract_path = '../../NewData/AllData/3.voxel/Train_'
checkpoint_model_path= "../../models/"
model_fn = 'voxel_cnn_lstm_100epochs.h5'
EPOCHS = 100
training_sample_cnt = 700                     # 한번에 학습할 데이터세트 크기 설정 Model Architecture와 GPU mermoy에 따라 실행 가능한 sample수 달라질 수 있음

sub_dirs=['Circle', 'Spread', 'Spin', 'ForwardBack', 'downToUp', 'upToDown', 'leftToRight', 'rightToLeft', 'diag-LeftToRight', 'diag-RightToLeft']

import gc
import os.path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv3D, MaxPooling3D, LSTM, GRU, Dense, Dropout, Flatten, Bidirectional, TimeDistributed, BatchNormalization
from sklearn.model_selection import train_test_split


def one_hot_encoding(y_data, sub_dirs, categories=5):
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


def full_3D_model(input_x, input_y, reg = 0, num_feat_map = 16, summary=False):
    print('building the model ... ')
    model = Sequential()
    # 1st layer group
    model.add(TimeDistributed(
        Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv1a", input_shape=(10, 32, 32, 1), padding="same",
               activation="relu")))
    # 2nd layer group
    # model.add(TimeDistributed(Conv3D(12, (3, 3, 3), strides=(1, 1, 1), name="conv1b", padding="same", activation="relu")))

    model.add(TimeDistributed(MaxPooling3D(name="pool1", strides=(2, 2, 2), pool_size=(2, 2, 2), padding="valid")))

    # 3rd layer group
    model.add(
        TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv2a", padding="same", activation="relu")))
    # model.add(TimeDistributed(Conv3D(12, (3, 3, 3), strides=(1, 1, 1), name="conv2b", padding="same", activation="relu")))
    model.add(TimeDistributed(
        MaxPooling3D(strides=(2, 2, 2), pool_size=(2, 2, 2), data_format="channels_first", name="pool2",
                     padding="valid")))

    model.add(
        TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv2a", padding="same", activation="relu")))
    # model.add(TimeDistributed(Conv3D(12, (3, 3, 3), strides=(1, 1, 1), name="conv2b", padding="same", activation="relu")))
    model.add(TimeDistributed(
        MaxPooling3D(strides=(2, 2, 2), pool_size=(2, 2, 2), data_format="channels_first", name="pool2",
                     padding="valid")))

    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.1))

    model.add(Bidirectional(LSTM(16, return_sequences=False, stateful=False)))
    # model.add(GRU(16, return_sequences=False, stateful=False))

    model.add(Dropout(.3))

    model.add(Dense(input_y.shape[1], activation='softmax', name='output'))

    return model


#train_data, train_label  = get_Training_Dataset()

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

    return train_data,train_label


def t_split(train_data,train_label): #지역변수 사용하여 함수 호출 종료 후 메모리 자동삭제 될 수 있도록 함수로 구현

    train_label = one_hot_encoding(train_label, sub_dirs, categories=5)
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], train_data.shape[3],
                                    train_data.shape[4], 1)
    print('Training Data Shape is:')
    print(train_data.shape, train_label.shape)
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.0001, random_state=1)


    print('>> X_train, y_train', X_train.shape, y_train.shape)
    print('>> X_val, y_val', X_val.shape, y_val.shape)

    return X_train, X_val, y_train, y_val


print('loading training dataset!!')
train_data,train_label  = dataset_load()
X_train, X_val, y_train, y_val = t_split(train_data,train_label) # shuffling을 위함인데 이부분은 어쩔 수 없이 데이터세터 크기의 2배 만큼 메모리 할당됨....
                                                                    # 메모리 부족한 경우 별도의 프로그램 코드를 통해 shuffling 한 후 읽어와야할 것 같음...
                                                                    # tensorflow dataset lib를 사용하는 것도 방법임.....메모리는 이 방법도 섞는 과정에서 2배 증가
                                                                    # dataset 이 가용 GPU memory보다 커 데 dataset을 분할하여 학습할 경우 적절한 shuffling은 필수 임.
                                                                    # shuffing하지 않고 dataset을 file에서 읽어온 순서로 분할하여 학습할 경우 학습률이 떨어짐.
del train_data,train_label #변수 삭제
gc.collect()  # 메모리 clear해줘야 깔끔하게 해당 변수의 memory 반환됨...


print('----------------------------------')


loop_cnt = len(X_train) // training_sample_cnt # training_sample_cnt로 나누어 학습했을 때 전체를 학습하기 위한 횟수 계산

for i in range(EPOCHS):   # GPU 메모리 부족하여 dataset 나눠서 전체데이터에 대해 10번 반복. 전체 데데이터를 학습한다는 관점에서 일반적인 학습에서 epoch 수로 사용
    print('loop ', i+1, '-------------------')
    for index in range(loop_cnt + 1): # 학습할 전체 dataset를 training_sample_cnt만큼 나누어 학습 진행
        ## GPU가 잡고 있는 메모리 해제,
        # 일반적으로 tensorflow에서는 연산에 필요한 memory 전체를 cpu->gpu 메모리로 할당함에 따라 dataset이 큰 경우 추가적인 GPU가 필요함
        # 그럴 수 없는 경우 학습하고 있는 session을 종료하고 할당된 memory를 해제하고 다시 할당하는 방식을 따르는데
        # memory segmentation문제 발생할 수는 있음.
        tf.keras.backend.clear_session() # 학습한 keras model session 종료
        gc.collect()                     # model session에 할당된 memory release GPU 메모라 해제할 때도 이걸 씀....
        ##-----------------------------------------------------------------------

        if os.path.exists(model_fn):   # 전이학습을 위해 저정된 모델이 있으면 읽어와서 weight update
            model = keras.models.load_model(model_fn)
            print(">> Model loaded")

        else:                            # 전이학습할 model이 없으면 모델 생성
            model = full_3D_model(X_train, y_train)
            opt_adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy'])
            print("Model building is completed")

        # model.fit에 넣어줄 X_train, y_train에 대한 split index 생성
        begin = index*training_sample_cnt      #시작 index
        end   = (training_sample_cnt*index)+training_sample_cnt #끝 index
        if index == loop_cnt:   # training_sample_cnt 단위로 나누었을 때 나머지 부분
            end = len(X_train)

        print('------------------------------------------------------------------------', index, begin, end)
        learning_hist = model.fit(X_train[begin:end], y_train[begin:end], batch_size=16, epochs=1, verbose=1, shuffle=True)

    model.save(model_fn)

print(model.summary())
