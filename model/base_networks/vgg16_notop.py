from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPool2D, Conv2D, Reshape, Concatenate, Activation, Input, ZeroPadding2D, MaxPooling2D
from tensorflow.python.keras.utils import data_utils

weights_notop_url = ('https://storage.googleapis.com/tensorflow/'
                     'keras-applications/vgg16/'
                     'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

class VGG16:
    def __init__(self, input_shape=None,
                 kernel_initializer=None,
                 kernel_regularizer=None):
        input_layer = Input(shape=input_shape, name="input")

        #block1
        conv1_1 = Conv2D(63, (3, 3), activation='relu', padding='same', name='block1_conv1',
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input_layer)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(conv1_1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', padding='same')(conv1_2)

        #block2
        conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(pool1)
        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(conv2_1)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', padding='same')(conv2_2)

        #block3
