import os
import os.path
import logging
import logging.handlers

import xml.etree.ElementTree as ET

import math
import cv2 as cv

class TrainingDataGenerator():

    ''' Generates training images for the tree detection pipeline '''

    predefined_classes_file = "predefined_classes.txt"
    log_file_path = "training_data_generator.log"

    # defining image transformation parameters
    resized_width = 320
    resized_height = 240

    def __init__(self, scr_folder, target_folder, block_dim=20, debug=False):

        ''' 
        Inputs
        ------
        src_folder (str) : folder containing the labelled data files and the original images
        target_folder (str) : folder in which to create training images
        block_dim (int) : size (in pixels) of the training images (square images) 
        debug (boolean) : True = display the training cutouts made from the manual segmentation
        '''

        self.src_folder = scr_folder
        self.target_folder = target_folder
        self.debug = debug

        # defining sub block sizes (square)
        self.block_dim = block_dim
        self.min_fill_area = 0.70 * self.block_dim**2

        # setting up logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_handler = logging.handlers.WatchedFileHandler(self.log_file_path)
        log_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(log_handler)

        # defining the predefined classes
        classes = []
        self.load_classes()


    def load_classes(self):

        ''' Loads classes from the predefined classes file'''

        try:
            with open(os.path.join(self.src_folder, self.predefined_classes_file), "r") as class_file_h:
                self.classes = class_file_h.readlines()
        except Exception as e:
            logging.error("TrainingDataGenerator failed to load the predefined classes  : {}".format(e))


    def generate_training_images(self):

        ''' Goes through all the annotation files and creates the proper training images in the target folder '''

        # going through the annotation data files in the scr folder
        for annotation_file in os.listdir(self.src_folder):
            if annotation_file.endswith(".xml"):

                try : 

                    # loading the current file xml
                    file_tree = ET.parse(os.path.join(self.src_folder, annotation_file)) 
                    file_root = file_tree.getroot()
        
                    # extracting dimension information
                    image_width = int(file_root.find('./size/width').text)
                    image_height = int(file_root.find('./size/height').text)
                    width_ratio = self.resized_width / image_width 
                    height_ratio = self.resized_height / image_height

                    # loading and resizing the referenced image
                    image_file_path = file_root.find('./path').text
                    image = cv.imread(image_file_path, cv.IMREAD_COLOR)
                    image = cv.resize(image, (self.resized_width, self.resized_height), 
                                      interpolation = cv.INTER_AREA)

                    block_index = 0

                    # going through the labelled objects
                    for labeled_object in file_root.findall('./object'):

                        # extracting the object label
                        object_label = labeled_object.find('./name').text

                        # extracting object positional information
                        xmin = int(int(labeled_object.find('./bndbox/xmin').text) * width_ratio)
                        ymin = int(int(labeled_object.find('./bndbox/ymin').text) * height_ratio)
                        xmax = int(int(labeled_object.find('./bndbox/xmax').text) * width_ratio)
                        ymax = int(int(labeled_object.find('./bndbox/ymax').text) * height_ratio)

                        # calculating the possible horizontal indicies
                        min_row_pos = (ymin // self.block_dim) * self.block_dim
                        n_vertical_blocks = ((math.ceil(ymax / self.block_dim) * self.block_dim) - min_row_pos) // self.block_dim
                        min_col_pos = (xmin // self.block_dim) * self.block_dim
                        n_horizontal_blocks = ((math.ceil(xmax / self.block_dim) * self.block_dim) - min_col_pos) // self.block_dim

                        # going through the block touching the object
                        current_row = min_row_pos 
                        current_col = min_col_pos
                        for row_i  in range(n_vertical_blocks):
                            for col_i in range(n_horizontal_blocks):

                                fill_width = 0
                                fill_height = 0
                                
                                # calculating the horizontal fill
                                if xmin > current_col and xmin < (current_col + self.block_dim): 
                                    fill_width = (current_col + self.block_dim) - xmin
                                elif xmax > current_col and xmax < (current_col + self.block_dim):
                                    fill_width = xmax - current_col
                                else :
                                    fill_width = self.block_dim

                                # calculating the vertical fill
                                if ymin > current_row and ymin < (current_row + self.block_dim):
                                    fill_height = (current_row + self.block_dim) - ymin
                                elif ymax > current_row and ymax < (current_row + self.block_dim):
                                    fill_height = ymax - current_row
                                else :
                                    fill_height = self.block_dim

                                # only processing blocks which contain enough fill area
                                if (fill_width * fill_height) > self.min_fill_area:                                

                                    if self.debug :
                                        
                                        # adding rectangle overlay on current block (for visualization)
                                        start_point = (current_col, current_row)
                                        end_point = (current_col + self.block_dim, current_row + self.block_dim)
                                        image = cv.rectangle(image, start_point, end_point, (255, 0, 0), 1)                                

                                        # displaying and wainting for user input
                                        cv.imshow('image', image)  
                                        cv.waitKey(0)

                                    else:

                                        # creating/saving a sub-image from the current block
                                        roi = image[current_row : current_row + self.block_dim, current_col : current_col + self.block_dim]
                                        image_file_segs = image_file_path.split("/")[-1].split(".")
                                        block_file_name = image_file_segs[0] + "_" + str(block_index) + "_" + object_label + "." + image_file_segs[-1]
                                        block_save_path = os.path.join(self.target_folder, block_file_name)
                                        cv.imwrite(block_save_path, roi)

                                        block_index += 1

                                # moving to the next horizontal block
                                current_col += self.block_dim

                            # moving down a row of blocks
                            current_col = min_col_pos
                            current_row += self.block_dim
                        
                        # destroying debug windows
                        if self.debug : 
                            cv.destroyAllWindows()

                except Exception as e:
                    logging.error("TrainingDataGenerator failed while extracting data from label file : {0}, error : {1}\
                                  ".format(annotation_file, e))
                    raise


class TrainingDataLoader():

    ''' 
    Enables the loading of labeled training data from the an image dataset 
    The labeled data is the images generated by the "TrainingDataGenerator" 
    '''

    image_formats = ["png", "jpg"]


    def __init__(self, image_folder, class_def_path):

        '''
        Parameters
        ----------
        image_folder (str) : path to the folder containing the training images
        class_def_path (str) : path to the class definition file
        '''

        self.image_folder = image_folder
        self.class_def_path = class_def_path

        self.class_map = {}
        self.load_classes()

    
    def load_classes(self):

        ''' Creates a class map from the predefined classes file '''

        with open(self.class_def_path, "r") as class_file_h:
            class_list = class_file_h.readlines()

        for class_i, class_name in enumerate(class_list):
            self.class_map[class_name.rstrip()] = class_i


    def load_training_data(self):

        ''' Generates the X (features) and y (label) lists for model training '''

        feature_container = []
        label_container = []

        # going through the training images
        for element in os.listdir(self.image_folder):
            if element.split(".")[-1] in self.image_formats:

                image_path = os.path.join(self.image_folder, element)
                
                image_mat = cv.imread(image_path, cv.IMREAD_COLOR)
                image_label = self.class_map[(image_path.split("_")[-1].split(".")[0])]
                feature_container.append(image_mat)
                label_container.append(image_label)
                
        return feature_container, label_container