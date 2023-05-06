from contentDetectron import detectron
import os
import argparse
import subprocess
import warnings

warnings.filterwarnings("ignore")


def parse_cli_arguments():
	parser = argparse.ArgumentParser(
		u"Python intro/autro detection for video files. Processes video inputs into segments: intro : outro"
	)
	parser.add_argument(u"--video_dir", required=True, help="video directory where the video files are stored")
	parser.add_argument(u"--feature_vector_method", nargs='?', const='CH', type=str, default='CH')
	parser.add_argument(u"--artifacts_dir", required=False, nargs='?', type=str, const='./artifacts/',
						default='./artifacts/')
	parser.add_argument(u"--framejump", nargs='?', const=8, type=int, default=8)
	parser.add_argument(u"--percentile", nargs='?', const=20, type=int, default=20)
	parser.add_argument(u"--resize_width", nargs='?', const=720, type=int, default=720)
	parser.add_argument(u"--end_threshold", nargs='?', const=7, type=int, default=7)
	parser.add_argument(u"--min_seconds", nargs='?', const=3, type=int, default=3)
	return parser.parse_args()


def main():
	command_params = parse_cli_arguments()
	video_folder = command_params.video_dir
	feature_vector_method = command_params.feature_vector_method
	artifacts_files = command_params.artifacts_dir
	framejump_val = command_params.framejump
	percentile_cutter = command_params.percentile
	resize_frame = command_params.resize_width
	end_threshold = command_params.end_threshold
	minimum_sec = command_params.min_seconds

	results = detectron.detect(video_dir=video_folder, feature_vector_function=feature_vector_method,
							   artifacts_dir=artifacts_files, framejump=framejump_val, percentile=percentile_cutter,
							   resize_width=resize_frame, video_end_threshold_seconds=end_threshold,
							   min_detection_size_seconds=minimum_sec)

	with open(os.path.join('./outputs', 'outputs_for_season.csv'), "w") as outputs:
		for key in results.keys():
			outputs.write("%s, %s\n"%(key, results[key]))

	print("Outputs saved into ./outputs/outputs_for_season.csv file.")


if __name__ == "__main__":
	main()
