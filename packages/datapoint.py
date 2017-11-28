import cv2
import numpy as np


class Datapoint(object):
    def __init__(self, center_path, left_path, right_path, angle, throttle, brake, speed):
        self._info = {'center_image': center_path,
                      'left_image': left_path,
                      'right_image': right_path,
                      'angle': angle,
                      'throttle': throttle,
                      'brake': brake,
                      'speed': speed}

    @property
    def center_image_path(self):
        return self._info['center_image']

    @property
    def left_image_path(self):
        return self._info['left_image']

    @property
    def right_image_path(self):
        return self._info['right_image']

    @property
    def steering_angle(self):
        return self._info['angle']

    @property
    def throttle(self):
        return self._info['throttle']

    @property
    def brake(self):
        return self._info['brake']

    @property
    def speed(self):
        return self._info['speed']

    def load_image(self, img_pos: str='center'):
        if img_pos not in ['center', 'left', 'right']:
            raise Exception('Image camera position not recognized, try center left or right.')

        img = None
        if img_pos == 'center':
            img = cv2.imread(self.center_image_path)

        if img_pos == 'left':
            img = cv2.imread(self.left_image_path)

        if img_pos == 'right':
            img = cv2.imread(self.right_image_path)

        if img is None:
            raise FileNotFoundError()

        return img

    def load_combined_image(self):
        img_left = self.load_image('left')
        img_right = self.load_image('right')
        img_center = self.load_image('center')

        img_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY)
        img_center = cv2.cvtColor(img_center, cv2.COLOR_RGB2GRAY)

        img = np.dstack([img_left, img_center, img_right])
        return img
