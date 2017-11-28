from packages.datapoint import Datapoint
import csv
import os
import ntpath
import numpy as np
import random as rnd


class SimulatorDatasetImporter(object):
    def __init__(self):
        self._dataset = []

    @property
    def dataset(self):
        return self._dataset

    def clear_dataset(self):
        self._dataset = []

    def append_dataset(self, csv_file_path, exclude_angles: list=None):
        if os.path.isfile(csv_file_path) is False:
            raise FileNotFoundError('Could not read csv in DatasetImporter.')

        csv_directory = ntpath.dirname(csv_file_path)
        csv_directory = os.path.join(csv_directory, 'IMG')
        with open(csv_file_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if exclude_angles is not None and float(row[3]) in exclude_angles:
                    continue

                datapoint = Datapoint(os.path.join(csv_directory, ntpath.basename(row[0])),
                                      os.path.join(csv_directory, ntpath.basename(row[1])),
                                      os.path.join(csv_directory, ntpath.basename(row[2])),
                                      float(row[3]),
                                      float(row[4]),
                                      float(row[5]),
                                      float(row[6]))
                self._dataset.append(datapoint)

    def harmonize_angles(self, epsilon=1e-1):
        # collect all angles in a dictionary
        angles_dict = {}
        for element in self._dataset:
            rounded = float(int(element.steering_angle / epsilon) * epsilon)
            if rounded not in angles_dict:
                angles_dict[rounded] = []
            # building up the histogram
            angles_dict[rounded].append(element)

        # Calc the maximum count of a rounded steering angle
        angle_max_count = np.max(np.array([len(angles_dict[k]) for k in angles_dict]))

        # Now, fill up a new dictionary with indices
        angles_dict_harmonize = angles_dict

        for k in angles_dict_harmonize:
            needed_for_fill = angle_max_count - len(angles_dict[k])
            for i in range(needed_for_fill):
                angles_dict_harmonize[k].append(rnd.choice(angles_dict[k]))

        # Overwrite dataset with harmonized version of itself
        self._dataset = []
        for k in angles_dict_harmonize:
            for element in angles_dict_harmonize[k]:
                self._dataset.append(element)
        # Done
        return
