from __future__ import print_function
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')
import warnings
warnings.filterwarnings("ignore")


def VGG16(input_shape, weights_path = None):
	img_input = Input(shape=input_shape)

	# Layer_1
	x = Convolution2D(64, 3, 3, activation='relu', padding='same', name='layer1_conv1')(img_input)
	x = Convolution2D(64, 3, 3, activation='relu', padding='same', name='layer1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='layer1_pool')(x)

	# Layer_2
	x = Convolution2D(128, 3, 3, activation='relu', padding='same', name='layer2_conv1')(x)
	x = Convolution2D(128, 3, 3, activation='relu', padding='same', name='layer2_conv2')(x)
	x = MaxPooling2D((2,2), strides=(2,2), name='layer2_pool')(x)

	# Layer_3
	x = Convolution2D(256, 3, 3, activation='relu', padding='same', name='layer3_conv1')(x)
	x = Convolution2D(256, 3, 3, activation='relu', padding='same', name='layer3_conv2')(x)
	x = Convolution2D(256, 3, 3, activation='relu', padding='same', name='layer3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2,2), name='layer3_pool')(x)

	# Layer_4
	x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='layer4_conv1')(x)
	x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='layer4_conv2')(x)
	x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='layer4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='layer4_pool')(x)

	# Layer_5
	x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='layer5_conv1')(x)
	x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='layer5_conv2')(x)
	x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='layer5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2,), name='layer5_pool')(x)

	# Layer_6 Classification task
	x = Flatten(name="flatten")(x)
	x = Dense(4096, activation='relu', name='fully_connected1')(x)
	x = Dense(4096, activation='relu', name='fully_connected2')(x)
	x = Dense(1000, activation='softmax', name='prediction')(x)

	# Packing the model
	model = Model(img_input, x)
	# Load pre-trained if available
	if weights_path:
		model.load_weights(weights_path)

	return model