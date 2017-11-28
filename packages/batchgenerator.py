import numpy as np
import keras.preprocessing.image as kgenerators


class BatchGenerator(kgenerators.Iterator):
    """ This class implements a simple batch generator. """

    def __init__(self,
                 batch_size,
                 n_classes,
                 dataset,
                 image_shape,
                 augmentation_fn=None,
                 preprocessing_fn=None,
                 extract_image_fn=None,
                 extract_label_fn=None,
                 shuffle=True,
                 seed=1337):

        self._x = dataset
        self._augmentation_fn = augmentation_fn
        self._preprocessing = preprocessing_fn

        if extract_image_fn is None:
            extract_image_fn = (lambda e: e.load_image())

        if extract_label_fn is None:
            extract_label_fn = (lambda e: e.steering_angle)

        self._extract_label_fn = extract_label_fn
        self._extract_image_fn = extract_image_fn

        self._image_shape = image_shape
        self._batch_size = batch_size
        self._num_classes = n_classes

        super().__init__(n=len(dataset), batch_size=batch_size, shuffle=shuffle, seed=seed)

    def _get_batches_of_transformed_samples(self, index_array):

        batch_x_elements = np.take(self._x, index_array)
        batch_y = [self._extract_label_fn(e) for e in batch_x_elements]
        batch_x = np.zeros(shape=[len(index_array),
                                  self._image_shape[0],
                                  self._image_shape[1],
                                  self._image_shape[2]])

        # Load images, preprocess and augment
        for i in range(len(batch_x_elements)):
            batch_x[i] = self._extract_image_fn(batch_x_elements[i])

            # Do augmentation if function is set
            if self._augmentation_fn is not None:
                batch_x[i], batch_y[i] = self._augmentation_fn(batch_x[i], batch_y[i])

            # Do preprocessing if function is set
            if self._preprocessing is not None:
                batch_x[i] = self._preprocessing(batch_x[i])

        assert len(batch_x) == len(batch_y)
        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)
