import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from collections import deque

from .ssd_utils import BBoxUtility


class Vehicle:

    def __init__(self, location, size, type):
        self._type = type
        self._location = location
        self._size = size
        self._detected = True

    @property
    def types(self):
        return set(['Car', 'Motorbike', 'Bicycle', 'Bus'])

    def accept(self, location, size):
        self._location = location
        self._size = size
        self._detected = True

    def reject(self):
        self._detected = False

class VehicleDetector:
    voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                   'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                   'Sheep', 'Sofa', 'Train', 'Tvmonitor']

    NUM_CLASSES = len(voc_classes) + 1

    def __init__(self, model, n_history=50):
        self._model = model
        self._history = deque([], n_history)
        self._bbox_util = BBoxUtility(self.NUM_CLASSES)

    @classmethod
    def draw_boxes(cls, img, results):
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

        colors = {
            'Car': (255, 0, 0),
            'Bus': (0, 0, 255),
            'Motorbike': (255, 255, 0),
            'Bicycle': (0, 255, 255)
        }

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = cls.voc_classes[label - 1]
            display_text = '{} [{:0.2f}]'.format(label_name, score)
            if label_name in set(('Car', 'Bus', 'Motorbike', 'Bicycle')):
                color = colors[label_name]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)
                cv2.putText(img, display_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1., color, 2)
        return img


    def video_pipeline(self):
        def process_video(input_img):
            inputs = []
            #input_img_cropped = input_img[120:720,680:1280,:]
            #img = cv2.resize(input_img_cropped, (300, 300))
            img = cv2.resize(input_img, (300, 300))
            img = image.img_to_array(img)
            inputs.append(img.copy())
            inputs = preprocess_input(np.array(inputs))
            inputs = np.expand_dims(inputs[0], axis=0)

            preds = self._model.predict(inputs, batch_size=1, verbose=0)
            results = self._bbox_util.detection_out(preds)

            #final_img_cropped = draw_boxes(input_img_cropped, results[0])
            #final_img = input_img.copy()
            #final_img[120:720,680:1280,:] = final_img_cropped
            final_img = self.draw_boxes(input_img, results[0])

            return final_img
        return process_video