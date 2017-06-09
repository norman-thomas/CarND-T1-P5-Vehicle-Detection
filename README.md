# CarND Project 5: Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.


# General Approach

For this project I decided to use a deep learning based approach as opposed to the classic computer vision approach using HOG (Histogram of Oriented Gradients) in combination with a Linear SVM classifier. I chose to try a DL based solution out of personal interest as well as out of performance reasons. Using a classifier on HOG features is faster than passing an image through a deep CNN, but the need for a sliding window heavily slows down the whole process as each image then requires up to a hundred patches to be cropped and classified. The CNN approach requires the image to be only processed once.

## Choice of Neural Network

There are several neural network architectures, which have been developed to solve the task of detecting object boundaries in an image. Most prominently, R-CNN, Fast-R-CNN, Faster-R-CNN, YOLO and SSD are known to yield good results. Comparing the speed and accuracy of these networks quickly reveals that YOLO and SSD are the fastest options, with SSD taking the lead. Accuracy-wise SSD has high scores. Out of these considerations I chose to apply SSD300 for this project.


# Single Shot Multibox Detector (SSD)

![SSD network](examples/SSD.png)





# References

[Towards a real-time vehicle detection: SSD multibox approach](https://chatbotslife.com/towards-a-real-time-vehicle-detection-ssd-multibox-approach-2519af2751c)
[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
[YOLO](https://pjreddie.com/darknet/yolo/)
[YOLOv2](https://arxiv.org/abs/1612.08242)
[YOLOv2 in Keras/TensorFlow](https://github.com/allanzelener/YAD2K)

---

[Evolving Boxes for Fast Vehicle Detection](https://arxiv.org/abs/1702.00254)
