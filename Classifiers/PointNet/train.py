
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from Classifiers.utils.datasetLoader import dataset_load

extract_path = '../../NewData/AllData/5.pointcloud/Train_'
model_fn = 'pointNet_full.h5'
checkpoint_model_path= "../../../models/"

MAX_EPOCHS = 500
sub_dirs=['Circle', 'Spread', 'Spin', 'ForwardBack', 'downToUp', 'upToDown', 'leftToRight', 'rightToLeft', 'diag-LeftToRight', 'diag-RightToLeft']
# sub_dirs=['Circle', 'Spread', 'Spin', 'ForwardBack', 'downToUp', 'upToDown', 'leftToRight', 'rightToLeft']

NUM_POINTS = 64
NUM_CHANNELS = 4
NUM_CLASSES = len(sub_dirs)
BATCH_SIZE = 32

# Dataset Load =====================================================================================================

train_data, train_label = dataset_load(extract_path, sub_dirs, onehot=False)

print('Training Data Shape is:')
print(train_data.shape,train_label.shape)

train_points, val_points, train_labels, val_labels  = train_test_split(train_data.astype(np.float64), train_label, test_size=0.20, random_state=1)
del train_data,train_label

# Points Augment & Shuffle ====================================================

train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((val_points, val_labels))

train_dataset = train_dataset.shuffle(len(train_points)).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(val_points)).batch(BATCH_SIZE)

# Create Model =====================================================================================================
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

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

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

inputs = keras.Input(shape=(NUM_POINTS, NUM_CHANNELS))


x = tnet(inputs, 4)

x = conv_bn(x, 64)
x = conv_bn(x, 64)

x = tnet(x, 64)

x = conv_bn(x, 64)
x = conv_bn(x, 128)
x = conv_bn(x, 1024)

x = layers.GlobalMaxPooling1D()(x)

x = dense_bn(x, 512)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

# Train =====================================================================================================

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=40, restore_best_weights=True)
model.fit(train_dataset, epochs=MAX_EPOCHS, validation_data=test_dataset, callbacks=[callback])
model.save(model_fn)


# Prediction =====================================================================================================

"""

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            sub_dirs[int(preds[i])], sub_dirs[int(labels[i])]
        )
    )
    ax.set_axis_off()
plt.show()

"""