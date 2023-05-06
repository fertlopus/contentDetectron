from __future__ import division
import numpy as np


def get_size_vgg_feature_map(input_w, input_h):
	output_w, output_h = input_w, input_h
	for i in range(1, 6):
		output_h = np.floor(output_h/2)
		output_w = np.floor(output_w/2)
	return output_w, output_h


def rmac_regions(width, height, length):
	# overlapping ration between neighbor regions
	ovr = 0.4
	# all possible regions for different dimensions
	steps = np.array([2, 3, 4, 5, 6, 7], dtype=np.float)

	w = min(width, height)
	b = (max(height, width) - width) / (steps - 1)
	# steps regions for rmac regions if dimension is more than 3
	idx = np.argmin(abs(((w**2 - w*b)/w**2)-ovr))

	# region per dimension
	Wd, Hd = 0, 0
	if height < width:
		Wd = idx + 1
	elif height > width:
		Hd = idx + 1

	regions = []

	for l in range(1, length+1):
		wl = np.floor(2*w/(l+1))
		wl2 = np.floor(wl/2 - 1)

		b = (width - wl) / (l + Wd - 1)
		if np.isnan(b):
			b = 0
		cenW = np.floor(wl2 + np.arange(0, l + Wd) * b) - wl2

		b = (height - wl) / (l + Hd - 1)
		if np.isnan(b):
			b = 0
		cenH = np.floor(wl2 + np.arange(0, l + Wd) * b) - wl2

		for i in cenH:
			for j in cenW:
				R = np.array([j, i, wl, wl], dtype=np.int)
				if not min(R[2:]):
					continue
				regions.append(R)
	regions = np.asarray(regions)
	return regions

