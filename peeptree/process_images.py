'''
Entry point for the testing of the Tree trunk recognition
'''

import os
import os.path

import cv2 as cv
from peeptree.processing import ImageProcessor

if __name__ == "__main__":

    # defining necessary paths
    image_extensions = ["jpg", "png"]
    trained_clf_path = "classifier.pickle"
    image_dir = "/home/one_wizard_boi/Documents/Projects/DJI-tree-detection/TrainingData/OriginalImages"

    # defining the image processor
    processor = ImageProcessor(trained_clf_path, block_size=20, debug=True)

    # going through the test images
    for element in os.listdir(image_dir):
        if element.split(".")[-1] in image_extensions:

            # loading and processing the target image
            image_path = os.path.join(image_dir, element)
            image = cv.imread(image_path, cv.IMREAD_COLOR)
            processor.detect_object_segments(image)