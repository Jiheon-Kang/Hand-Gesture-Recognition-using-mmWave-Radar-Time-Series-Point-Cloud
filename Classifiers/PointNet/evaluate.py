
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import gc

from sklearn.model_selection import train_test_split
from Classifiers.utils.datasetLoader import dataset_load
from keras_flops import get_flops

extract_path = '../../NewData/AllData/5.pointcloud/Test_'
model_fn = 'pointnet_full.h5'

sub_dirs=['Circle', 'Spread', 'Spin', 'ForwardBack', 'downToUp', 'upToDown', 'leftToRight', 'rightToLeft', 'diag-LeftToRight', 'diag-RightToLeft']

NUM_POINTS = 64
NUM_CHANNELS = 4
NUM_CLASSES = len(sub_dirs)


# ==================================================================================


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    #def get_config(self):
    #    return {"num_features": self.num_features, "l2reg": self.l2reg}

    def get_config(self):
        config = {"num_features": self.num_features, "l2reg": self.l2reg}
        return config

# Dataset Load =====================================================================================================

test_points, test_labels = dataset_load(extract_path, sub_dirs, onehot=False)

print('Test Data Shape is:')
print(test_points.shape,test_labels.shape)

# test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
# test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

# Evaluate =========================================================================================================

print("Load Model:", model_fn)

new_model = keras.models.load_model(model_fn, custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})
# new_model = keras.models.load_model(model_fn)

new_model.summary()
flops = get_flops(new_model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 6: .03} M")
# print(f"FLOPS: {flops / 10 ** 9: .03} G")

tf.keras.backend.clear_session()  # 학습한 keras model session 종료
gc.collect()  # model 학습에 할당된 memory release

print("-- Evaluate --")
test_loss, test_acc = new_model.evaluate(test_points, test_labels,  verbose=2)
print(test_loss, test_acc)

import time

test = test_points[0].reshape((1,)+test_points[0].shape)
print("-- Predict --")
before_time = time.time()
y_pred = new_model.predict(test, 1)
after = time.time()
print("inference time", after - before_time)
print(test.shape)

# Confusion Metrix =================================================================================================
print("-- Confusion Metrix --")

print(test_points.shape[0])
y_pred = new_model.predict(test_points, 1)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

y_true = test_labels.squeeze()
y_pred = np.argmax(y_pred, axis=1)
print(y_true.shape, y_pred.shape)

cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sub_dirs)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.title('pointNet')
plt.tight_layout()
plt.show()
