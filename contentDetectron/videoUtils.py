import cv2
import ffmpeg


def get_framerate(video_file):
	video = cv2.VideoCapture(video_file)
	return video.get(cv2.CAP_PROP_FPS)


def resize(video_file, outfile, resize_width):
	video = cv2.VideoCapture(video_file)
	frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

	if frame_count > 0:
		stream = ffmpeg.input(video_file)
		if resize_width == 224:
			stream = ffmpeg.filter(stream, 'scale', w=244, h=244)
		else:
			# in order to return the same aspect ratio during resizing the h = trunc(ow/a/2)*2
			# more here ---> https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
			stream = ffmpeg.filter(stream, 'scale', w=resize_width, h="trunc(ow/a/2)*2")
		stream = ffmpeg.output(stream, outfile)
		try:
			ffmpeg.run(stream)
		except FileNotFoundError:
			raise Exception("ffmpeg is not found on your device, install ffmpeg")
	else:
		raise Exception(f"The video file {video_file} provided is not supported or corrupted.")
