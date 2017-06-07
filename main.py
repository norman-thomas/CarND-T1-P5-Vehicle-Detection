#!/usr/bin/python3

from moviepy.editor import VideoFileClip

from helpers.ssd import SSD300
from helpers.ssd_utils import BBoxUtility
from helpers.utils import VehicleDetector


if __name__ == '__main__':
    input_shape=(300, 300, 3)
    model = SSD300(input_shape, num_classes=VehicleDetector.NUM_CLASSES)
    model.load_weights('./weights_SSD300.hdf5', by_name=True)
    detector = VehicleDetector(model)

    output = 'project_video_SSD.mp4'
    clip = VideoFileClip('project_video.mp4')
    clip = clip.subclip(t_start=35, t_end=44)
    clip = clip.fl_image(detector.video_pipeline())
    clip.write_videofile(output, audio=False)
