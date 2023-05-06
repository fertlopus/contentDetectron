import os
import itertools
import numpy as np
import faiss
import operator
import pickle
import datetime
from natsort import natsorted, ns
from . import featureVectorizer
from . import videoUtils
from . import evaluation


def max_two_values(d):
	v = list(d.values())
	k = list(d.keys())
	result1 = k[v.index(max(v))]
	del d[result1]
	v = list(d.values())
	k = list(d.keys())
	result2 = k[v.index(max(v))]
	return [result1, result2]


def fill_gaps(sequence, lookahead):
	"""
	Fills the gap in features sequence in case of the gap between 1's and 0's values.
	:param sequence: list consisting of 0's and 1's
	:param lookahead: skipping params in sequence
	:return: list of sequence filled where gap has occured
	"""
	i = 0
	change_needed = False
	look_left = 0
	while i < len(sequence):
		look_left -= 1
		if change_needed and look_left < 1:
			change_needed = False
		if sequence[i]:
			if change_needed:
				for k in to_change:
					sequence[k] = True
			else:
				change_needed = True
			look_left = lookahead
			to_change = []
		else:
			if change_needed:
				to_change.append(i)
		i += 1
	return sequence


def get_two_longest_timestamps(timestamps):

	if len(timestamps) <= 2:
		return timestamps

	d = {}
	for start, end in timestamps:
		d[(start, end)] = end - start

	return max_two_values(d)


def to_time_string(seconds):
	return str(datetime.timedelta(seconds=seconds))


def query_episodes_with_faiss(videos, vectors_dir):

	vector_files = [os.path.join(vectors_dir, e+'.p') for e in videos]
	vectors = []
	lengths = []

	for f in vector_files:
		episode_vectors = np.array(pickle.load(open(f, 'rb')), np.float32)
		lengths.append(episode_vectors.shape[0])
		vectors.append(episode_vectors)

	vectors = np.vstack(vectors)
	results = []

	for i, length in enumerate(lengths):
		print(f"Querying the video file identified as {videos[i]}")
		i += 1
		s = sum(lengths[:i - 1])
		e = sum(lengths[:i])
		query = vectors[s:e]
		rest = np.append(vectors[:s], vectors[e:], axis=0)
		vector_size = query.shape[1]
		index = faiss.IndexFlatL2(vector_size)
		index.add(rest)
		k = 1
		scores, indexes = index.search(query, k)
		result = scores[:, 0]
		results.append((videos[i - 1], result))
	return results


def detect(video_dir, feature_vector_function = 'CH', annotations = None, artifacts_dir = None, framejump = 3,
		   percentile = 10, resize_width = 320, video_start_threshold_percentile = 20, video_end_threshold_seconds = 15,
		   min_detection_size_seconds = 15):

	if feature_vector_function == 'CNN':
		resize_width = 224
	print(f"Detection started...\nFramejump: {framejump}\nVideo Width: {resize_width}\nFeature Vector Type: {feature_vector_function}")
	resized_dir_name = f"resized{resize_width}"
	feature_vector_dir_name = f"{feature_vector_function}_feature_vectors_framejump{framejump}"
	videos = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
	videos = natsorted(videos, alg=ns.IGNORECASE)

	if artifacts_dir is None:
		artifacts_dir = video_dir

	vectors_dir = os.path.join(artifacts_dir, resized_dir_name, feature_vector_dir_name)

	if annotations is not None:
		annotations = evaluation.get_annotations(annotations)

	for file in videos:
		file_full = os.path.join(video_dir, file)
		file_resized = os.path.join(artifacts_dir, resized_dir_name, file)
		os.makedirs(os.path.dirname(file_resized), exist_ok=True)

		if not os.path.isfile(file_resized):
			print(f"Resizing {file}")
			videoUtils.resize(file_full, file_resized, resize_width)

		print(f"Converted {file} to feature vectors")
		featureVectorizer.construct_feature_vectors(file_resized, feature_vector_dir_name, feature_vector_function,
													framejump)
	results = query_episodes_with_faiss(videos, vectors_dir)
	total_relevant_sec = 0
	total_detected_sec = 0
	total_relevant_detected_sec = 0
	all_detected = {}

	for video, result in results:
		framerate = videoUtils.get_framerate(os.path.join(video_dir, video))
		threshold = np.percentile(result, percentile)
		below_threshold = result < threshold
		below_threshold = fill_gaps(below_threshold, int((framerate / framejump) * 10))
		nonzeros = [[i for i, v in it] for k, it in itertools.groupby(enumerate(below_threshold),
																	  key=operator.itemgetter(1))
					if k != 0]
		detected_beginning = []
		detected_end = []

		for nonzero in nonzeros:
			start = nonzero[0]
			end = nonzero[-1]
			occurs_at_beggining = end < len(result) * (video_start_threshold_percentile/100)
			ends_at_the_end = end  > len(result) - video_end_threshold_seconds * (framerate/framejump)

			if (end - start > (min_detection_size_seconds * (framerate/framejump))
				and (occurs_at_beggining or ends_at_the_end)):
				start = start/(framerate/framejump)
				end = end / (framerate/framejump)
				if occurs_at_beggining:
					detected_beginning.append((start, end))
				elif ends_at_the_end:
					detected_end.append((start, end))

		detected = get_two_longest_timestamps(detected_beginning) + detected_end

		print(f"Detection for {video}")

		for s, e in detected:
			print(f"{to_time_string(s)} \t \t - \t \t {to_time_string(e)}")

		if annotations is not None:
			ground_truth = evaluation.skip_timestamps_in_file(video, annotations)
			relevant_sec, detected_sec, relevant_detected_sec = evaluation.precision_recall_detections_score(
				detected, ground_truth
			)
			total_relevant_sec += relevant_sec
			total_detected_sec += detected_sec
			total_relevant_detected_sec += relevant_detected_sec
		all_detected[video] = detected

	if annotations is not None:
		precision = total_relevant_detected_sec / total_detected_sec
		recall = total_relevant_detected_sec / total_relevant_sec
		print(f"Precision: {precision} ----- Recall: {recall}")
	return all_detected
