"""
This package contains all code needed for training on simulator datasets
"""

from packages.dataset import SimulatorDatasetImporter
from packages.batchgenerator import BatchGenerator
from packages.imageaugmentation import ImageAugmenter
from packages.kerasmodels import squeeze_net, nvidia_net, mobile_net

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda, Input, Cropping2D
from keras.optimizers import Adam, SGD

import numpy as np
import cv2
import os

# In the csv file there are the following fields:
# Center image, left image, right image, steering angle, throttle, brake, speed
importer = SimulatorDatasetImporter()
importer_validation = SimulatorDatasetImporter()

importer.append_dataset('../traindata/old_trainingdata/driving_crazy/driving_log.csv', exclude_angles=[0.0])
# importer.append_dataset('../traindata/driving/driving_log.csv', exclude_angles=[])
# importer.append_dataset('../traindata/driving_reverse/driving_log.csv', exclude_angles=[])
# importer.append_dataset('../traindata/driving_normal_datarun0/driving_log.csv', exclude_angles=[])

importer.append_dataset('../traindata/1_lap_forward_track1_1_lap_forward_track2/driving_log.csv', exclude_angles=[])
importer.append_dataset('../traindata/1_lap_backward_track1/driving_log.csv', exclude_angles=[])
importer.append_dataset('../traindata/1_lap_backward_track2/driving_log.csv', exclude_angles=[])
importer.harmonize_angles(epsilon=1e-2,
                          exclude_angles=[],
                          show_histogram=False,
                          exclude_less_than=2,
                          random_sample_max_to=1000,
                          center=False)

#importer_validation.append_dataset('../traindata/validation_drive/driving_log.csv', exclude_angles=[])

importer_validation.append_dataset('../traindata/1_lap_forward_track1_val/driving_log.csv', exclude_angles=[])
importer_validation.append_dataset('../traindata/1_lap_forward_track2_val/driving_log.csv', exclude_angles=[])
importer.harmonize_angles(epsilon=1e-2,
                          exclude_angles=[],
                          show_histogram=False,
                          exclude_less_than=None,
                          center=False)

# Define the cropping positions
crop_top = 60
crop_bottom = 20

# Get first image for shape
img = importer.dataset[0].load_combined_image()

# Training hyperparameter
experiment_name = 'MobileNet_TraindataSetV2_crazy'

batch_size = 64
epochs = 10

augmenter = ImageAugmenter()

augmenter.add_median(prob=0.05, k=3)
#augmenter.add_invert(prob=0.1)
augmenter.add_coarse_dropout(prob=0.05, size_percentage=0.02)
augmenter.add_gaussian_noise(prob=0.05, scale=10)
augmenter.add_simplex_noise(prob=0.05, multiplicator=0.8)
#augmenter.add_invert(prob=0.1)
augmenter.add_keras_augmenter(ImageDataGenerator(rotation_range=2.,
                                                 width_shift_range=0.02,
                                                 height_shift_range=0.02,
                                                 zoom_range=0.02,
                                                 fill_mode='constant'))

preprocessing = (lambda x: np.expand_dims(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), -1))

batch_gen = BatchGenerator(batch_size=batch_size,
                           n_classes=1,
                           extract_xy_fn=[(lambda e: e.load_image_and_label('center', np.random.randint(20, 30)/100.)),
                                          (lambda e: e.load_image_and_label('left', np.random.randint(20, 30)/100.)),
                                          (lambda e: e.load_image_and_label('right', np.random.randint(20, 30)/100.))
                                          ],
                           dataset=importer.dataset,
                           augmentation_fn=augmenter.augment,
                           preprocessing_fn=preprocessing,
                           image_shape=img.shape)

batch_gen_validation = BatchGenerator(batch_size=batch_size, n_classes=1,
                                      dataset=importer_validation.dataset,
                                      preprocessing_fn=preprocessing,
                                      image_shape=img.shape)

img = batch_gen_validation.custom_next()[0][0]
batch_gen_validation.reset()

log_dir = os.path.join('logs', experiment_name)
if os.path.isdir(log_dir) is False:
    os.makedirs(log_dir)

if False:
    for e in range(1):
        imgs, y = batch_gen.custom_next()
        for j, img in enumerate(imgs):
            img = img[60: img.shape[0]-20]
            #cv2.imwrite('img_aug' + str(j) + '.png', img)
            cv2.imshow('win', img.astype(np.uint8))
            cv2.waitKey(2000)

in_layer = Input(shape=img.shape)
in_layer = Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)))(in_layer)
in_layer = Lambda(lambda in_img: ((in_img - 128.) / 128.))(in_layer)

filter_multiplicator = 1
x = nvidia_net(nb_classes=1, filter_multiplicator=filter_multiplicator, input_shape=None, dropout=0.2, input_tensor=in_layer)
# x = squeeze_net(nb_classes=1, input_shape=None, dropout=0.2, input_tensor=in_layer)
# x = mobile_net(nb_classes=1, filter_multiplicator=filter_multiplicator, input_shape=None, dropout=0.2, input_tensor=in_layer)

x.summary()
#x.load_weights('../NvidiaNet_V6_BS256_ExtremeAugfiltermul0.25_weights.01-0.00797.hdf5')

x.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

experiment_name = experiment_name + \
                  '[' + str(filter_multiplicator)+']'+\
                  '[BS'+str(batch_size)+']'

callbacks = [TensorBoard(log_dir),
             ModelCheckpoint(filepath=experiment_name + "_weights.{epoch:02d}-{val_loss:.5f}.hdf5",
                             monitor='loss',
                             mode='min')]

steps_per_epoch = int(np.ceil(len(importer.dataset) / batch_size))
steps_per_epoch_val = int(np.ceil(len(importer_validation.dataset) / batch_size))
print('Steps per Epoch: ', steps_per_epoch)
print('Steps per Validation Epoch: ', steps_per_epoch_val)

x.fit_generator(callbacks=callbacks,
                verbose=1,
                generator=batch_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=batch_gen_validation,
                validation_steps=steps_per_epoch_val,
                workers=12)
x.save('trained_model.h5')
