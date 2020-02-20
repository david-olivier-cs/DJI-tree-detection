import os
import os.path

import cv2 as cv
import numpy as np
from .model import TreeClassifier


class DetectedObject():

    ''' Coordinate representation of objects in an image '''

    def __init__(self, top_left, bottom_right):
        
        '''
        Parameters
        ----------
        top_left (int, int) : top left bounding box coordinates
        bottom_right (int, int) : bottom right bounding box coordinates
        '''

        self.top_left = top_left
        self.bottom_right = bottom_right


class ImageProcessor():

    ''' Applies the necessary processing steps for tree recognition '''

    # defining the label for detected object segments
    detected_segment_label = 1


    def __init__(self, clf_path, block_size, resized_width=320, resized_height=240, debug=False):

        '''
        Parameters
        ----------
        clf_path (str) : path to the pickled classifier to use
        '''

        self.debug = debug
        self.block_size = block_size
        self.resized_width = resized_width
        self.resized_height = resized_height

        # defining pixels increments
        self.n_blocks_col = self.resized_width // self.block_size
        self.n_blocks_row = self.resized_height // self.block_size


        # defining classifier for object recognition
        self.clf = TreeClassifier(clf_path)


    def process_frame(self, image):

        '''
        Parameters
        ------
        image (numpy.ndarray) : Image in 3D color space (RBG or HSV)
        
        Returns
        -------
        list(DetectedObject) : list of detected objects
        '''

        # resizing the input image
        image = cv.resize(image, (self.resized_width, self.resized_height), 
                          interpolation = cv.INTER_AREA)

        object_segments = []

        # going through the blocks of the input image
        for row_i in range(self.n_blocks_row):
            
            seg_row_start = row_i * self.block_size

            for col_i in range(self.n_blocks_col):

                # extracting current subimage
                seg_col_start = col_i * self.block_size     
                image_seg = image[seg_row_start : seg_row_start + self.block_size, seg_col_start : seg_col_start + self.block_size]
                image_seg = np.expand_dims(image_seg, axis=0)

                # collecting object segments
                if self.clf.predict(image_seg) == self.detected_segment_label:
                    seg_top_left = (seg_col_start, seg_row_start)
                    seg_bottom_right = (seg_col_start + self.block_size, seg_row_start + self.block_size)
                    object_segments.append(DetectedObject(seg_top_left, seg_bottom_right))

        # debug mode displays detected object segments
        if self.debug:
            for obj_seg in object_segments:
                image = cv.rectangle(image, obj_seg.top_left, obj_seg.bottom_right, (255, 0, 0), 1)                                
            cv.imshow("Detected segments", image)  
            cv.waitKey(0)







