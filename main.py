#!/usr/bin/python3

import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
#from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from helpers.ssd import SSD300
from helpers.ssd_utils import BBoxUtility

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']

NUM_CLASSES = len(voc_classes) + 1

input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('./weights_SSD300.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

def draw_boxes(img, results):
    # Parse the outputs.
    det_label = results[:, 0]
    det_conf = results[:, 1]
    det_xmin = results[:, 2]
    det_ymin = results[:, 3]
    det_xmax = results[:, 4]
    det_ymax = results[:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = voc_classes[label - 1]
        display_txt = '{}'.format(label_name)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        if label_name == 'Car':
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 5)
            #currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            #currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    return img


def process_video(input_img):
    inputs = []
    # input_img_cropped = input_img[120:720,680:1280,:]
    # img = cv2.resize(input_img_cropped, (300, 300))
    img = cv2.resize(input_img, (300, 300))
    img = image.img_to_array(img)
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))
    inputs = np.expand_dims(inputs[0], axis=0)

    preds = model.predict(inputs, batch_size=1, verbose=0)
    results = bbox_util.detection_out(preds)

    # final_img_cropped = draw_boxes(input_img_cropped, preds, results)
    # final_img = input_img
    # input_img[120:720,680:1280,:] = final_img_cropped
    final_img = draw_boxes(input_img, results[0])

    return final_img

output = 'project_video_video_SSD.mp4'
clip1 = VideoFileClip('project_video.mp4')
clip = clip1.fl_image(process_video)
clip.write_videofile(output, audio=False)
