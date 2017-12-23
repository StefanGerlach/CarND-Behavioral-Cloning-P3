from keras import models
from keras.layers import Input, Dropout, Activation, AveragePooling2D, GlobalMaxPooling2D, Flatten, Dense,\
    GlobalAveragePooling2D, BatchNormalization, Concatenate, Conv2D, Reshape, MaxPool2D, SeparableConv2D

from keras.engine import get_source_inputs
import numpy as np

def fire_module(x, filters, dropout):
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Conv2D(sq_filters, (1, 1), activation='relu', padding='same')(x)
    expand1 = Conv2D(ex1_filters, (1, 1), activation='relu', padding='same')(squeeze)
    expand2 = Conv2D(ex1_filters, (3, 3), activation='relu', padding='same')(squeeze)
    x = Concatenate(axis=-1)([expand1, expand2])
    return x


# https://arxiv.org/abs/1602.07360
def squeeze_net(nb_classes, input_shape, dropout, input_tensor=None):
    model_input = Input(input_shape) if input_tensor is None else input_tensor

    x = BatchNormalization()(model_input)
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D()(x)

    x = fire_module(x, (16, 64, 64), dropout)
    x = fire_module(x, (16, 64, 64), dropout)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = fire_module(x, (32, 128, 128), dropout)
    x = fire_module(x, (32, 128, 128), dropout)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = fire_module(x, (48, 192, 192), dropout)
    x = fire_module(x, (48, 192, 192), dropout)

    x = fire_module(x, (64, 256, 256), dropout)
    x = fire_module(x, (64, 256, 256), dropout)

    x = Conv2D(192, kernel_size=(1, 1), activation='relu')(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(32, activation='relu')(x)
    x = Dense(nb_classes, activation='sigmoid')(x)

    squeezenet = models.Model(get_source_inputs(model_input), x)
    return squeezenet


def nvidia_net(nb_classes, filter_multiplicator, input_shape, dropout, input_tensor=None):
    model_input = Input(input_shape) if input_tensor is None else input_tensor

    x = Conv2D(int(np.max([24*filter_multiplicator, 1])), kernel_size=(5, 5), padding='same')(model_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(int(np.max([36*filter_multiplicator, 1])), kernel_size=(5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(int(np.max([48*filter_multiplicator, 1])), kernel_size=(5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(int(np.max([64*filter_multiplicator, 1])), kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(int(np.max([64*filter_multiplicator, 1])), kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalMaxPooling2D()(x)
    x = Dropout(dropout)(x)

    x = Dense(100, activation='relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(50, activation='relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(10, activation='relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(nb_classes)(x)

    nvidia_net = models.Model(get_source_inputs(model_input), x)
    return nvidia_net


# https://arxiv.org/abs/1704.04861
def mobile_net(nb_classes, filter_multiplicator, input_shape, dropout, input_tensor=None):
    model_input = Input(input_shape) if input_tensor is None else input_tensor

    # 1. Standard Convolution 2D strides = 2, 2
    x = Conv2D(int(np.max([32*filter_multiplicator, 1])),
               kernel_size=(3, 3),
               strides=(2, 2),
               activation='relu',
               padding='same')(model_input)
    x = BatchNormalization()(x)

    # 2. Depthwise separable convolution
    x = SeparableConv2D(int(np.max([64*filter_multiplicator, 1])),
                        kernel_size=(3, 3),
                        activation='relu',
                        padding='same')(x)
    x = BatchNormalization()(x)

    # 3. Depthwise separable convolution strides = 2, 2
    x = SeparableConv2D(int(np.max([128 * filter_multiplicator, 1])),
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        activation='relu',
                        padding='same')(x)
    x = BatchNormalization()(x)

    # 4. Depthwise separable convolution strides = 1, 1
    x = SeparableConv2D(int(np.max([128 * filter_multiplicator, 1])),
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation='relu',
                        padding='same')(x)
    x = BatchNormalization()(x)

    # 5. Depthwise separable convolution strides = 2, 2
    x = SeparableConv2D(int(np.max([256 * filter_multiplicator, 1])),
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        activation='relu',
                        padding='same')(x)
    x = BatchNormalization()(x)

    # 6. Depthwise separable convolution strides = 1, 1
    x = SeparableConv2D(int(np.max([256 * filter_multiplicator, 1])),
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation='relu',
                        padding='same')(x)
    x = BatchNormalization()(x)

    # 7. Depthwise separable convolution strides = 2, 2
    x = SeparableConv2D(int(np.max([512 * filter_multiplicator, 1])),
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        activation='relu',
                        padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalMaxPooling2D()(x)
    x = Dropout(dropout)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(nb_classes, activation='sigmoid')(x)

    mobilenet = models.Model(get_source_inputs(model_input), x)
    return mobilenet
