"""
This package contains all code needed for training on simulator datasets
"""

from packages.dataset import SimulatorDatasetImporter
from packages.batchgenerator import BatchGenerator
from packages.imageaugmentation import ImageAugmenter
from packages.kerasmodels import squeeze_net

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda, Input, Cropping2D
from keras.optimizers import Adam, SGD

import numpy as np
import cv2

# In the csv file there are the following fields:
# Center image, left image, right image, steering angle, throttle, brake, speed
importer = SimulatorDatasetImporter()
# Dont Train this crazy stuff !!
# importer.append_dataset('../traindata/driving_crazy/driving_log.csv', exclude_angles=[0.0])
importer.append_dataset('../traindata/driving/driving_log.csv', exclude_angles=[])
importer.append_dataset('../traindata/driving_reverse/driving_log.csv', exclude_angles=[])
importer.harmonize_angles()

# Define the cropping positions
crop_top = 60
crop_bottom = 20

# Get first image for shape
img = importer.dataset[0].load_combined_image()

# Training hyperparameter
batch_size = 64
epochs = 50

augmenter = ImageAugmenter()

augmenter.add_coarse_dropout()
augmenter.add_simplex_noise(multiplicator=0.2)
augmenter.add_keras_augmenter(ImageDataGenerator(rotation_range=5.,
                                                 width_shift_range=0.02,
                                                 height_shift_range=0.02,
                                                 zoom_range=0.05,
                                                 fill_mode='constant'))

batch_gen = BatchGenerator(batch_size=batch_size, n_classes=1, dataset=importer.dataset,
                           augmentation_fn=augmenter.augment, image_shape=img.shape)

callbacks = []
callbacks.append(TensorBoard())
callbacks.append(ModelCheckpoint(filepath="weights.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss',
                                 mode='min'))


if False:
    for e in range(100):
        imgs, y = batch_gen.next()
        for img in imgs:
            img = img[60: img.shape[0]-20]
            cv2.imshow('win', img.astype(np.uint8))
            cv2.waitKey(100)

in_layer = Input(shape=img.shape)
in_layer = Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)))(in_layer)
in_layer = Lambda(lambda in_img: (in_img-128.) / 128.)(in_layer)

x = squeeze_net(nb_classes=1, input_shape=None, input_tensor=in_layer)
x.summary()
x.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

steps_per_epoch = int(np.ceil(len(importer.dataset) / batch_size))
print('Steps per Epoch: ', steps_per_epoch)

x.fit_generator(callbacks=callbacks, verbose=1, generator=batch_gen, steps_per_epoch=1, epochs=epochs, workers=8)
x.save('trained_model.h5')
