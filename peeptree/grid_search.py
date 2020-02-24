''' Entry point script for performing parameter gridsearchs '''

import json
import pickle

from peeptree.model import TreeClassifierKNN, TreeClassifierSVM
from peeptree.data import TrainingDataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score

from sklearn.externals.joblib import parallel_backend

if __name__ == "__main__":

    # defining the necessary paths
    results_path_template = "search_results_{}.pickle"
    param_grid_path = "grid_search_params.json"
    class_definitions_path = "predefined_classes.txt"
    training_folder_prefix = "/home/one_wizard_boi/Documents/Projects/DJI-tree-detection/TrainingData/LabeledData_"

    # loading the grid search params
    with open(param_grid_path) as grid_file_h:
        param_grid = json.load(grid_file_h)

    # performing grid searches for every input image size
    image_sizes = [15, 20, 25, 30]
    for image_size in image_sizes:

        print("\n\nPerforming grid search with image size : ({0} X {0})\n\n".format(str(image_size)))

        # loading training data
        training_folder_path = training_folder_prefix + str(image_size)
        data_loader = TrainingDataLoader(training_folder_path, class_definitions_path)
        X, y = data_loader.load_training_data()

        # defining the grid search
        clf_pipeline = TreeClassifierSVM.classification_pipeline()
        g_search = GridSearchCV(estimator=clf_pipeline, param_grid=param_grid["svm"], 
                                scoring="recall", n_jobs=3, cv=3, verbose=10)
        
        # performing the grid search and exporting the results
        with parallel_backend('threading'):
            g_search.fit(X, y)

        with open(results_path_template.format(str(image_size)), 'wb') as handle:
            pickle.dump(g_search.cv_results_, handle)