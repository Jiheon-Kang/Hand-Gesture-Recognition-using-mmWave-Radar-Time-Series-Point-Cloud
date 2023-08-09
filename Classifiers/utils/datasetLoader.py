import numpy as np

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

def getIntLabel(y_data, sub_dirs):
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
    y_features=y_features.reshape(y_features.shape[0],1).astype(int)

    return y_features

def read_datasetFile(extract_path, labelname):
    Data_path = extract_path+labelname
    data = np.load(Data_path+'.npz')
    # float64를 float32로 줄임 : 메모리 최적화를 위해....
    return np.array(data['arr_0'], dtype=np.dtype(np.float32)), data['arr_1']


def dataset_load(extract_path, sub_dirs, onehot=True): # 학습 데이터 파일에서 읽자!~ 지역변수 사용하여 함수 호출 종료 후 메모리 자동삭제 될 수 있도록 함수로 구현
    train_data, train_label = read_datasetFile(extract_path, sub_dirs[0])
    print('1= ', train_data.shape,'accumulate : ',train_label.shape)

    for idx, sub_dir in enumerate(sub_dirs[1:]):
        temp_train_data, temp_train_label = read_datasetFile(extract_path, sub_dir)
        train_data = np.concatenate((train_data, temp_train_data), axis=0)
        train_label = np.concatenate((train_label, temp_train_label), axis=0)
        print(str(idx+2)+'= ', temp_train_data.shape, 'accumulate : ', train_label.shape)
    if onehot:
        train_label = one_hot_encoding(train_label, sub_dirs)
    else:
        train_label = getIntLabel(train_label, sub_dirs)
    return train_data,train_label