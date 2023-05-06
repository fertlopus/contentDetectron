from __future__ import division, print_function
import warnings
import cv2
from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.models import Model
import keras.backend as K
from keras.applications.vgg16 import VGG16 # as keras has own implementation of VGG16 the own implementation is no longer needed
from .RoiPooling import RoiPooling
from .get_regions import rmac_regions, get_size_vgg_feature_map
from .utils import *
import scipy.io
import numpy as np


K.set_image_dim_ordering('th')
warnings.filterwarnings("ignore")


INPUT_DIMENSION = (224, 224)
vector_size = 512


def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, vector_size, axis=-1)
    out = x * w
    return out


def rmac(input_shape, num_rois):
    vgg16_model = VGG16(input_shape = input_shape, weights = 'imagenet', include_top=False)
    in_roi = Input(shape=(num_rois, 4), name='input_roi')
    x = RoiPooling([1], num_rois)([vgg16_model.layers[-5].output, in_roi])
    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norml2')(x)
    # PCA
    x = TimeDistributed(Dense(vector_size, name='pca', kernel_initializer='identity',
                              bias_initializer='zeros'))(x)
    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)
    # Addition
    rmac = Lambda(addition, output_shape=(vector_size,), name='rmac')(x)
    # Rmac normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)
    # Model
    model = Model([vgg16_model.input, in_roi], rmac_norm)
    # Load weights
    mat = scipy.io.loadmat(DATA_DIR + PCA_FILE)
    b = np.squeeze(mat['bias'], axis=1)
    w = np.transpose(mat['weights'])
    model.layers[-4].set_weights([w, b])
    return model


model = None
regions = None


def load_model():
    global regions
    global model
    Wmap, Hmap = get_size_vgg_feature_map(INPUT_DIMENSION[0], INPUT_DIMENSION[1])
    regions = rmac_regions(Wmap, Hmap, 3)
    model = rmac((3, INPUT_DIMENSION[0], INPUT_DIMENSION[1]), len(regions))


def to_feature_vector(img_array):
    if model is None:
        load_model()
    img = cv2.resize(img_array, dsize=INPUT_DIMENSION, interpolation=cv2.INTER_NEAREST)
    img = img.reshape((3, INPUT_DIMENSION[0], INPUT_DIMENSION[1]))
    img = np.expand_dims(img, axis=0)
    RMAC = model.predict([img, np.expand_dims(regions, axis=0)])
    return RMAC[0, :].astype('float32')


