import cv2
import numpy as np
import os
import json
import pickle
from math import sqrt
from tqdm import tqdm
from . import rmac


def get_frame(frame_index, video):
	video.set(1, frame_index)
	_, img = video.read()
	return img


fouriers = [
	[1, 1, 1, 1, 1, 1, 1, 1],
	[-1, 1, -1, 1, 1, -1, 1, -1],
	[-sqrt(2) / 2, 0, sqrt(2) / 2, -1, 1, -sqrt(2) / 2, 0, sqrt(2) / 2],
	[-sqrt(2) / 2, -1, -sqrt(2) / 2, 0, 0, sqrt(2) / 2, 1, sqrt(2) / 2],
	[0, -1, 0, 1, 1, 0, -1, 0],
	[1, 0, -1, 0, 0, -1, 0, 1],
	[sqrt(2) / 2, 0, -sqrt(2) / 2, -1, 1, sqrt(2) / 2, 0, -sqrt(2) / 2],
	[-sqrt(2) / 2, 1, -sqrt(2) / 2, 0, 0, sqrt(2) / 2, -1, sqrt(2) / 2]
]

for i, f in enumerate(fouriers):
	f.insert(4, 0)
	fouriers[i] = np.array(f)
	fouriers[i] = fouriers[i].reshape((3, 3)).astype('float32')

max_vals = []
for f in fouriers:
	m = np.array([255])
	m = cv2.matchTemplate(m.astype('float32'), f, cv2.TM_CCORR).clip(0, 255)
	max_vals.append(cv2.matchTemplate(m.astype('float32'), f, cv2.TM_CCORR)[0][0])


def color_texture_moments(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	result = []
	for channel in range(0, 3):
		for template, max_val in zip(fouriers, max_vals):
			r = cv2.matchTemplate(img[:, :, channel].astype('float32'), template, cv2.TM_CCORR)
			r = r / max_val
			result.append(r.mean())
			result.append(r.std())

	return result


def cnn_feature_vectors(img):
	feature_vector = rmac.rmac.to_feature_vector(img)
	return feature_vector


def get_img_color_hist(img, binsize):
	channels = cv2.split(img)
	main = np.zeros((0, 1))
	# channels iteration process
	for channel in channels:
		hist = cv2.calcHist([channel], [0], None, [binsize], [0, 256])
		main = np.append(main, hist)
	# normalization
	main = main / (img.shape[0] * img.shape[1])
	return main.astype('float32')


def color_hist(img):
	result = get_img_color_hist(img, 100)
	return result


def construct_feature_vectors(video_filename, result_dir_name, vector_function, framejump):
	base_video_fn = os.path.basename(video_filename)
	video = cv2.VideoCapture(video_filename)
	series_dir = os.path.dirname(video_filename)
	vectors_filename = os.path.join(series_dir, result_dir_name, base_video_fn + ".p")

	# Method of feature vectorizing to apply:
	if vector_function == 'CH':
		vector_function = color_hist
	elif vector_function == 'CTM':
		vector_function = color_texture_moments
	elif vector_function == 'CNN':
		vector_function = cnn_feature_vectors

	os.makedirs(os.path.dirname(vectors_filename), exist_ok=True)
	if not os.path.isfile(vectors_filename):
		feature_vectors = []
		total = int(video.get(cv2.CAP_PROP_FRAME_COUNT) / framejump) - 1
		for i in tqdm(range(total)):
			img = get_frame(i * framejump, video)
			feature_vector = vector_function(img)
			feature_vectors.append(feature_vector)
		with open(vectors_filename, 'wb') as outfile:
			pickle.dump(feature_vectors, outfile, protocol=2)
