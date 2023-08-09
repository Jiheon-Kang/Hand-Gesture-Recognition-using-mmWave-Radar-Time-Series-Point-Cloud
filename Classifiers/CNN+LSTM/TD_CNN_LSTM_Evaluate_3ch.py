# TD_CNN_LSTM 코드에서 학습된  model을 불러와 test set에 대한 평가를 수행하는 code

import keras
import numpy as np
import tensorflow as tf
import gc
from tensorflow.keras.utils import to_categorical

# testset file 경로
extract_path = '../NewData/xyzProj/3channel/Test_'
model_path ='./model.h5'

sub_dirs=['Circle', 'DiagonalLeftToRight', 'DiagonalRightToLeft', 'FistForward', 'FistSpread', 'LeftRight', 'UpToDown']
sub_dirs_kor = ['원그리기', '대각선 왼오', '대각선 오왼', '주먹 앞으로', '주먹 펼치기', '좌우', '위아래']

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

new_model = keras.models.load_model(model_path)
new_model.summary()
tf.keras.backend.clear_session()  # 학습한 keras model session 종료
gc.collect()  # model 학습에 할당된 memory release

test_loss, test_acc = new_model.evaluate(X_test,  Y_test, verbose=2)
print(test_loss, test_acc)

print("-- Predict --")
y_pred = new_model.predict(X_test, 1)

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
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sub_dirs_kor)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.title('10 epochs')
plt.tight_layout()
plt.show()
