import pickle
import cv2 as cv
import numpy as np

from sklearn import svm
from skimage import feature
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


class TreeClassifierKNN(Classifier):

    ''' 
    Enables tree trunk classification as implemented in the 
    "Visual Tree Detection for Autonomous Navigation in Forest Environment" paper. 
    ''' 

    pipeline_steps = ["feature_extractor", "knn"]


    @classmethod
    def classification_pipeline(cls, **kwargs):

        ''' Returns an untrained classification pipeline'''

        # preparing configuration containers
        configs_container = []
        for _ in range(len(cls.pipeline_steps)):
            configs_container.append({})
        
        # seperating pipeline step parameters
        for step_i, pipeline_step in enumerate(cls.pipeline_steps):
            for config_param in kwargs.keys():
                if pipeline_step in config_param:
                    clean_param_name = config_param.split("__")[-1]
                    configs_container[step_i][clean_param_name] = kwargs[config_param]

        return Pipeline([
            ("feature_extractor", ImageFeatureExtractor(**configs_container[0]))
            ("knn", KNeighborsClassifier(**configs_container[1]))
        ])
        

class TreeClassifierSVM(Classifier):

    ''' Enables tree trunk classification with a SVM ''' 

    pipeline_steps = ["feature_extractor", "svm"]


    @classmethod
    def classification_pipeline(cls, **kwargs):

        ''' Returns an untrained classification pipeline'''

        # preparing configuration containers
        configs_container = []
        for _ in range(len(cls.pipeline_steps)):
            configs_container.append({})
        
        # seperating pipeline step parameters
        for step_i, pipeline_step in enumerate(cls.pipeline_steps):
            for config_param in kwargs.keys():
                if pipeline_step in config_param:
                    clean_param_name = config_param.split("__")[-1]
                    configs_container[step_i][clean_param_name] = kwargs[config_param]
        
        return Pipeline([
            ("feature_extractor", ImageFeatureExtractor(**configs_container[0])),
            ("svm", svm.SVC(**configs_container[1]))
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

        
    def fit(self, X, y=None, **kwargs):
        return self

    
    def transform(self, X, y=None):

        '''
        Parameters
        ------
        X (numpy.ndarray (4D)) : array of images in 3D color space (RBG or HSV)
        
        Returns
        -------
        (np.ndarray (2D)) : list of feature vectors
        '''

        # defining the output container
        feature_container = None

        # going through the input feature list
        for data_i, x in enumerate(X):

            # isolating image color channels
            img_channel_1 = x[..., 0]
            img_channel_2 = x[..., 1]
            img_channel_3 = x[..., 2]

            # extracting color channel histogram features (color features)
            color_vector_1 = self.compute_channel_histogram(img_channel_1)
            color_vector_2 = self.compute_channel_histogram(img_channel_2)
            color_vector_3 = self.compute_channel_histogram(img_channel_3)
            feature_vector = np.concatenate([color_vector_1, color_vector_2, color_vector_3])

            # extracting LBP features from gray scale image
            if self.fusion_method == 1:

                if self.color_space == "RGB":
                    feature_vector = np.concatenate([feature_vector, 
                                    self.compute_lbp_descriptor(cv.cvtColor(x, cv.COLOR_BGR2GRAY))])
                else: 
                    feature_vector = np.concatenate([feature_vector, self.compute_lbp_descriptor(img_channel_3)])

            # extracting LBP features from all color channels
            else:
                lbp_vector_1 = self.compute_lbp_descriptor(img_channel_1)
                lbp_vector_2 = self.compute_lbp_descriptor(img_channel_2)
                lbp_vector_3 = self.compute_lbp_descriptor(img_channel_3)
                feature_vector = np.concatenate([feature_vector, lbp_vector_1, lbp_vector_2, lbp_vector_3])

            # adding the feature vector to the ouput container
            if feature_container is None:
                feature_container = feature_vector
            else: feature_container = np.vstack((feature_container, feature_vector))

        # when a single feature vector is generated
        if feature_container.ndim == 1:
            feature_container = np.expand_dims(feature_container, axis=0)

        return feature_container


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