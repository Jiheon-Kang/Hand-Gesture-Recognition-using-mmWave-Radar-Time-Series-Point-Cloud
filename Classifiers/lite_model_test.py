import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

extract_path = '../NewData/230418/proj/Train_'
sub_dirs = ['LeftRight', 'UpDown']
model_path = 'TCN/cnn_tcn.tflite'
label_cnt = 2

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

test_X, test_Y = dataset_load()
test_X = test_X.reshape(-1, 27, 32, 32, 1)
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print(input_shape, input_details[0]['index'])

input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_details[0]['shape'])
print('data_shape: ', test_X.shape, test_Y.shape)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_details[0]['shape'])

test_X = test_X.astype(input_type)
test_X = test_X[:50]
test_Y = test_Y[:50]
data_len = len(test_X)
outputs = np.zeros((data_len, label_cnt))
for i in range(data_len):
    if(i%50==0):
        print(i)
    interpreter.set_tensor(input_details[0]['index'], test_X[i].reshape(1,27,32,32,1))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    outputs[i] = output_data[0]
# print(outputs.shape)
outputs = np.argmax(outputs, axis =-1)
test_Y = np.argmax(test_Y, axis =-1)
right_cnt = (outputs==test_Y).sum()
acc = float(right_cnt)/float(test_X.shape[0])
print('acc:', acc)
# acc: 0.841726618705036