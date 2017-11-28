from keras import models
from keras.layers import Input, Dropout,  AveragePooling2D, Dense,\
    GlobalAveragePooling2D, BatchNormalization, Concatenate, Conv2D, Reshape, MaxPool2D

from keras.engine import get_source_inputs


def fire_module(x, filters):
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Conv2D(sq_filters, (1, 1), activation='relu', padding='same')(x)
    expand1 = Conv2D(ex1_filters, (1, 1), activation='relu', padding='same')(squeeze)
    expand2 = Conv2D(ex1_filters, (3, 3), activation='relu', padding='same')(squeeze)
    x = Concatenate(axis=-1)([expand1, expand2])
    return x


def squeeze_net(nb_classes, input_shape, input_tensor=None):
    model_input = Input(input_shape) if input_tensor is None else input_tensor

    x = BatchNormalization()(model_input)
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D()(x)

    x = fire_module(x, (16, 64, 64))
    x = fire_module(x, (16, 64, 64))
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = fire_module(x, (32, 128, 128))
    x = fire_module(x, (32, 128, 128))
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = fire_module(x, (48, 192, 192))
    x = fire_module(x, (48, 192, 192))

    x = fire_module(x, (64, 256, 256))
    x = fire_module(x, (64, 256, 256))

    x = Conv2D(192, kernel_size=(1, 1), activation='relu')(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(32, activation='relu')(x)
    x = Dense(nb_classes)(x)

    squeezenet = models.Model(get_source_inputs(model_input), x)
    return squeezenet
