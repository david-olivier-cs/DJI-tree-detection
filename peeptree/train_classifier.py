''' Entry point script for training the tree classifier using the generated training data '''

import os
import json
import pickle
import os.path

from peeptree.data import TrainingDataLoader
from peeptree.model import TreeClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, recall_score


if __name__ == "__main__":

    # defining necessary paths
    output_model_path = "classifier.pickle"
    pipeline_config_path = "pipeline_params.json"
    class_definitions_path = "predefined_classes.txt"
    training_folder_prefix = "/home/one_wizard_boi/Documents/Projects/DJI-tree-detection/TrainingData/LabeledData_"

    # loading the wanted pipeline parameters
    with open(pipeline_config_path) as config_file_h:
        clf_params = json.load(config_file_h)

    # setting up the loading of training data
    training_folder_path = training_folder_prefix + str(clf_params["input_img_size"])
    data_loader = TrainingDataLoader(training_folder_path, class_definitions_path)

    # defining the untrained classification pipeline
    clf_pipeline = TreeClassifier.classification_pipeline(**clf_params)

    # loading training data
    X, y = data_loader.load_training_data()
    training_df = pd.DataFrame({'label': y})
    print("\nFeature set shape : ", X.shape, "\n")
    print("Label distribution :\n", training_df["label"].value_counts(), "\n")

    # checking model performance with cross validation
    scoring = {'accuracy': 'accuracy', 'recall': 'recall', 'precision': 'precision'}
    cross_val_scores = cross_validate(clf_pipeline, X, y, cv=3, scoring=scoring)
    print("cross validation scores : \n\n", cross_val_scores)

    # training and exporting the model
    clf_pipeline.fit(X,y)
    with open(output_model_path, 'wb') as handle:
        pickle.dump(clf_pipeline, handle)