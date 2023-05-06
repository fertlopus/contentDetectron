import os
import pandas as pd
import math


def count_overlap(first_interval, second_interval):
	return max(0, min(first_interval[1], second_interval[1]) - max(first_interval[0], second_interval[0]))


def timestamps_summation(timestamps):
	total_timestamps = 0
	for start, end in timestamps:
		total_timestamps += end - start
	return total_timestamps


def precision_recall_detections_score(detected, ground_truth, verbose=False):
	if verbose:
		print(f"Processing the results...\nDetected: \t \t {detected}\nGround truth: \t \t {ground_truth}")

	total_relevant_time_seconds = timestamps_summation(ground_truth)
	total_detected_time_seconds = timestamps_summation(detected)
	relevant_detected_time_seconds = 0

	for start, end in ground_truth:
		lowest_difference_index = 0
		lowes_difference = -1
		for i, (start_d, end_d) in enumerate(detected):
			if abs(start - start_d) < 2:
				start_d = start
			if abs(end - end_d) < 2:
				end_d = end
			relevant = count_overlap((start, end), (start_d, end_d))
			relevant_detected_time_seconds += relevant

	if verbose:
		# The output format will be in the form of: #of Relevant videos ---- #of Retrieved videos ---- # of Relevant and Retrieved videos
		print(f"Total relevant time,s : {total_relevant_time_seconds}\nTotal detected time,s : {total_detected_time_seconds}\nRelevant detected time,s : {relevant_detected_time_seconds}")
		if total_detected_time_seconds > 0:
			print(f"Precision score: {relevant_detected_time_seconds/total_detected_time_seconds}")
		if total_relevant_time_seconds > 0:
			print(f"Recall score: {relevant_detected_time_seconds/total_relevant_time_seconds}")
	return total_relevant_time_seconds, total_detected_time_seconds, relevant_detected_time_seconds


def merge_timestamps(timestamps):
	result = []
	i = 0
	while i < len(timestamps):
		(start, end) = timestamps[i]
		if i < len(timestamps) - 1:
			(start_next, end_next) = timestamps[i+1]
			if abs(end - start_next) < 2:
				result.append((start, end_next))
				i+=1
			else:
				result.append((start, end))
		else:
			result.append((start, end))
		i+=1
	return result


def convert_to_sec(time):
	if time is None:
		return -1
	try:
		hours, minutes, seconds = int(time.split(":")[0]), int(time.split(":")[1]), int(float(time.split(":")[2]))
		return hours*60*60 + minutes*60 + seconds
	except:
		if math.isnan(time):
			return -1


def skip_timestamps_in_file(filename, df):
	result = []
	try:
		row = df.loc[df['filename'] == filename].to_dict(orient='records')[0]

		if not row["recap_start"] == -1:
			result.append((row["recap_start"], row["recap_end"]))

		if not row["openingcredits_start"] == -1:
			result.append((row["openingcredits_start"], row["openingcredits_end"]))

		if not row["preview_start"] == -1:
			result.append((row["preview_start"], row["preview_end"]))

		if not row["closingcredits_start"] == -1:
			result.append((row["closingcredits_start"], row["closingcredits_end"]))
	except:
		raise Exception(f"The file {filename} is not supported or corrupted.")

	return merge_timestamps(result)


def get_annotations(filename):
	annotations = pd.read_csv(filename).dropna(how="all")
	# the beauty building :)
	annotations["recap_start"] = annotations["recap_start"].apply(convert_to_sec)
	annotations["recap_end"] = annotations["recap_end"].apply(convert_to_sec)
	annotations["openingcredits_start"] = annotations["openingcredits_start"].apply(convert_to_sec)
	annotations["openingcredits_end"] = annotations["openingcredits_end"].apply(convert_to_sec)
	annotations["preview_start"] = annotations["preview_start"].apply(convert_to_sec)
	annotations["preview_end"] = annotations["preview_end"].apply(convert_to_sec)
	annotations["closingcredits_start"] = annotations["closingcredits_start"].apply(convert_to_sec)
	annotations["closingcredits_end"] = annotations["closingcredits_end"].apply(convert_to_sec)
	return annotations
