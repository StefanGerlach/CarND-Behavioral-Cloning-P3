from packages.datapoint import Datapoint
import csv
import os
import ntpath
import numpy as np
import random as rnd
import matplotlib.pyplot as plt


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

    def harmonize_angles(self, epsilon=1e-1, exclude_angles: list=None, show_histogram=False):
        # collect all angles in a dictionary
        angles_dict = {}
        for element in self._dataset:
            rounded = float(int(element.steering_angle / epsilon) * epsilon)
            if rounded not in angles_dict:
                angles_dict[rounded] = []
            # building up the histogram
            angles_dict[rounded].append(element)

        # Exclude some angles
        if exclude_angles is not None:
            for angle_to_ex in exclude_angles:
                angles_dict.pop(angle_to_ex)

        if show_histogram:
            self.visualize_dataset_frequencies(angles_dict, 'Non-normalized steering angles')

        # Calc the maximum count of a rounded steering angle
        angle_max_count = np.max(np.array([len(angles_dict[k]) for k in angles_dict]))

        # Now, fill up a new dictionary with indices
        angles_dict_harmonize = angles_dict.copy()

        for k in angles_dict_harmonize:
            needed_for_fill = angle_max_count - len(angles_dict[k])
            for i in range(needed_for_fill):
                angles_dict_harmonize[k].append(rnd.choice(angles_dict[k]))

        # Overwrite dataset with harmonized version of itself
        self._dataset = []
        for k in angles_dict_harmonize:
            for element in angles_dict_harmonize[k]:
                self._dataset.append(element)

        if show_histogram:
            self.visualize_dataset_frequencies(angles_dict_harmonize, 'Normalized steering angles')
        # Done
        return

    def visualize_dataset_frequencies(self, y, title: str):
        # count the frequencies of classes in dataset and visualize
        hist = {}

        for label_id in sorted(y.keys()):
            hist[label_id] = len(y[label_id])

        # visualize as histogram
        fig = plt.figure(figsize=(16, 12))
        sub = fig.add_subplot(1, 1, 1)
        sub.set_title(title)
        y_data = np.array([float(hist[k]) for k in hist])
        plt.bar(range(len(hist)), y_data, align='center')
        x_axis = np.array([k for k in hist])
        plt.xticks(range(len(hist)), x_axis, rotation='vertical', fontsize=8)
        plt.subplots_adjust(bottom=0.4)
        plt.show()
