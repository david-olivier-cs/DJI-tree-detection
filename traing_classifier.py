'''
Entry point script for training the tree classifier using the generated training data
'''

import os
import os.path

from peeptree.data import TrainingDataLoader
from peeptree.model import TreeClassifier


if __name__ == "__main__":

    # setting up the loading of training data
    training_img_folder = "/home/one_wizard_boi/Documents/Projects/DJI-tree-detection/TrainingData/LabeledData_20"
    class_definitions_path = "/home/one_wizard_boi/Documents/Projects/DJI-tree-detection/predefined_classes.txt"
    data_loader = TrainingDataLoader(training_img_folder, class_definitions_path)

    # loading training data
    X, y = data_loader.load_training_data()
