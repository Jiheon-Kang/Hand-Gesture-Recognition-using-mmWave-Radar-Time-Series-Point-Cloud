# TD_CNN_LSTM 코드에서 학습된  model을 불러와 test set에 대한 평가를 수행하는 code

import keras
from tcn import TCN, tcn_full_summary
import numpy as np
import tensorflow as tf
import gc
from tensorflow.keras.utils import to_categorical
from keras_flops import get_flops
import time

# testset file 경로
extract_path = '../../NewData/AllData/4.proj/Test_'
# model_save_subfix = extract_path.replace('../../NewData/forTCN/','')
# model_save_subfix = model_save_subfix.replace('/Test_','')

model_path = 'proj_tcn.h5'

sub_dirs=['Circle', 'Spread', 'Spin', 'ForwardBack', 'downToUp', 'upToDown', 'leftToRight', 'rightToLeft', 'diag-LeftToRight', 'diag-RightToLeft']

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
    y_features = to_categorical(y_features)

    return y_features


#loading the train data
def read_datasetFile(labelname):
    Data_path = extract_path+labelname
    data = np.load(Data_path+'.npz')
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


X_test, Y_test = dataset_load()
# X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], X_test.shape[2], X_test.shape[3], 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], X_test.shape[2]*X_test.shape[3]*1)

new_model = keras.models.load_model(model_path, custom_objects={'TCN': TCN})
new_model.summary()
flops = get_flops(new_model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9: .03} G")

tf.keras.backend.clear_session()  # 학습한 keras model session 종료
gc.collect()  # model 학습에 할당된 memory release

test_loss, test_acc = new_model.evaluate(X_test,  Y_test, verbose=2)
print(test_loss, test_acc)

test = X_test[0].reshape((1,)+X_test[0].shape)
print("-- Predict --")
before_time = time.time()
y_pred = new_model.predict(test, 1)
after = time.time()
print("inference time", after - before_time)
print(test.shape)
"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

y_true = np.argmax(Y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sub_dirs)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.title('CNN+TCN with projection image - max epochs 200, early stopped')
plt.tight_layout()
plt.show()
"""