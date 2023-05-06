import pickle
import os

realpath = os.path.dirname(os.path.relpath(__file__))

DATA_DIR = realpath + '/data'
PCA_FILE = "PCAmatrices.mat"
IMG_SIZE = 1024


def save_obj(obj, filename):
	f = open(filename, 'wb')
	pickle.dump(obj, f)
	f.close()
	print(f"Saved to {filename}")


def load_obj(filename):
	f = open(filename, 'rb')
	obj = pickle.load(f)
	f.close()
	print(f"Loaded from {filename}")


def preprocess_image(x):
	# Normalization
	x[:, 0, :, :] -= 103
	x[:, 1, :, :] -= 116
	x[:, 2, :, :] -= 123

	# RgB -> BGR
	x = x[:, ::-1, :, :]
	return x
