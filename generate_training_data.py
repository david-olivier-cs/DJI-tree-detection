'''
Entry point script for generating training images from the original segmented images 
stored in the "SegmentedImages" folder.
'''

import os
import shutil
import os.path

from peeptree.data import TrainingDataGenerator

if __name__ == "__main__":

    # defining folder config
    src_folder = "/home/one_wizard_boi/Documents/Projects/DJI-tree-detection/TrainingData/SegmentedImages"
    target_folder_prefix = "/home/one_wizard_boi/Documents/Projects/DJI-tree-detection/TrainingData/LabeledData_{}"

    # defining the different image sizes to generate
    image_sizes = [15, 20, 25, 30]

    for image_size in image_sizes:

        # creating the target folder
        target_folder = target_folder_prefix.format(str(image_size))
        if os.path.isdir(target_folder):
            shutil.rmtree(target_folder)
        os.mkdir(target_folder)

        print("Generating training images with dimensions : {0} X {0}.".format(str(image_size)))

        # generating training data for current format
        data_generator = TrainingDataGenerator(src_folder, target_folder, block_dim=image_size)
        data_generator.generate_training_images()