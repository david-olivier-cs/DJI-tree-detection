from peeptree.data import TrainingDataGenerator

if __name__ == "__main__":

    # defining folder config
    src_folder = "/home/one_wizard_boi/Documents/Projects/DJI-tree-detection/TrainingData/SegmentedImages"
    target_folder = "/home/one_wizard_boi/Documents/Projects/DJI-tree-detection/TrainingData/LabeledData_20"

    data_generator = TrainingDataGenerator(src_folder, target_folder)
    data_generator.generate_training_images()