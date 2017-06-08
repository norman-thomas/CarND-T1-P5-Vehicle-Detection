#!/usr/bin/python3

import glob
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip

from helpers.ssd import SSD300
from helpers.utils import VehicleDetector

def process_test_images(detector):
    filenames = glob.glob('test_images/*.jpg')
    images = [np.asarray(Image.open(filename)) for filename in filenames]
    results = [detector.detect(img) for img in images]
    for i, img in enumerate(results):
        image = Image.fromarray(img)
        image.save('output_images/test_{}.jpg'.format(i), 'JPEG')

if __name__ == '__main__':
    input_shape=(300, 300, 3)
    model = SSD300(input_shape, num_classes=VehicleDetector.NUM_CLASSES)
    model.load_weights('./weights_SSD300.hdf5', by_name=True)
    detector = VehicleDetector(model)

    process_test_images(detector)

    output = 'project_video_SSD.mp4'
    clip = VideoFileClip('project_video.mp4')
    clip = clip.subclip(t_start=35, t_end=44)
    clip = clip.fl_image(detector.pipeline)
    clip.write_videofile(output, audio=False)
