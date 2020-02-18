import pickle
import cv2 as cv
import numpy as np

from skimage import feature
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class Classifier():

    '''
    Base class for clasifiers
    Loading a pre-trained model and exposing a predict function
    '''

    def __init__(self, classfier_path):
        
        '''
        Params
        ------
        classfier_path : str
            path to the classfier pickle file
        '''

        with open(classfier_path, 'rb') as f_handle:
            self.clf = pickle.load(f_handle)


    def predict(self, X):

        '''
        Returns
        -------
        int : predicted class
        '''

        return self.clf.predict(X)[0]


class TreeClassifier(Classifier):

    ''' 
    Enables tree trunk classificatino as implemented in the 
    "Visual Tree Detection for Autonomous Navigation in Forest Environment" paper. 
    ''' 

    def predict(self, X):
        return self.clf.predict()

    @staticmethod
    def classification_pipeline():

        ''' Returns an untrained classification pipeline'''

        return Pipeline([
            ("feature_extractor", ImageFeatureExtractor()),
            ("pca", PCA()),
            ("knn", KNeighborsClassifier())
        ])


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):

    ''' 
    Extracts features from provided images
    
    Features : 
        - Color histogram bins
        - LPB descriptor
    '''
    
    eps=1e-7
    color_spaces = ["RGB", "HSV"]
    
    # defining channel histogram dimensions
    channel_hist_n_bins = 15

    def __init__(self, color_space="HSV", lbp_n_points=8, lbp_radius=1, fusion_method=1):
        
        '''
        Parameters
        ----------
        lbp_radius (int) : neighbor radius used when calculating the lbp descriptor
        lbp_n_points (int) : number of neighbour considered when calculating the LBP values
        fusion_method (int) : 1 or 2, fusion methods a described in the reference paper
        color_space (str) : input image color space (HSV, RBG)
        '''

        if not color_space in self.color_spaces:
            raise ValueError("Invalid color space")

        if not (fusion_method == 1 or fusion_method == 2):
            raise ValueError("Invalid fusion method") 

        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.color_space = color_space
        self.fusion_method = fusion_method

        
    def fit(self, X, y=None):
        return self

    
    def transform(self, X, y=None):

        '''
        Parameters
        ------
        X (numpy.ndarray) : Image in 3D color space (RBG or HSV)
        
        Returns
        -------
        (np.ndarray) : feature vector
        '''

        # isolating image color channels
        img_channel_1 = X[..., 0]
        img_channel_2 = X[..., 1]
        img_channel_3 = X[..., 2]

        # extracting color channel histogram features (color features)
        color_vector_1 = self.compute_channel_histogram(img_channel_1)
        color_vector_2 = self.compute_channel_histogram(img_channel_2)
        color_vector_3 = self.compute_channel_histogram(img_channel_3)
        feature_vector = np.concatenate([color_vector_1, color_vector_2, color_vector_3])

        # extracting LBP features from gray scale image
        if self.fusion_method == 1:

            if self.color_space == "RGB":
                feature_vector = np.concatenate([feature_vector, 
                                 self.compute_lbp_descriptor(cv.cvtColor(X, cv.COLOR_BGR2GRAY))])
            else: 
                feature_vector = np.concatenate([feature_vector, self.compute_lbp_descriptor(img_channel_3)])

        # extracting LBP features from all color channels
        else:

            lbp_vector_1 = self.compute_lbp_descriptor(img_channel_1)
            lbp_vector_2 = self.compute_lbp_descriptor(img_channel_2)
            lbp_vector_3 = self.compute_lbp_descriptor(img_channel_3)
            feature_vector = np.concatenate([feature_vector, lbp_vector_1, lbp_vector_2, lbp_vector_3])

        return feature_vector


    def compute_lbp_descriptor(self, gray_img):

        ''' Computes the LBP decriptor for the provided gray scale image '''

        # computing the LBP histogram
        lbp = feature.local_binary_pattern(gray_img, self.lbp_n_points, self.lbp_radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_n_points + 3), 
                                 range=(0, self.lbp_n_points + 2))

        # normalizing the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + self.eps)
        
        return hist


    def compute_channel_histogram(self, channel_img):

        ''' Computes a normalized value histogram for the provided single channel image '''

        (hist, _) = np.histogram(channel_img.ravel(), bins=self.channel_hist_n_bins)
        hist = hist.astype("float")
        hist /= (hist.sum() + self.eps)
        return hist