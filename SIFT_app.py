#!/usr/bin/env python

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np
import time

""" 
Author: Christian Malherbe

App for showing homography for an image, and a video feed. I don't have a webcam, so I recorded
a video and am using that recording.


"""

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)

        self.vid = "/home/fizzer/SIFT_app/Robot Video.mp4"
        self.VidCap = cv2.VideoCapture(self.vid)

    	self.ImgPath = "/home/fizzer/SIFT_app/000_image.jpg"


    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        #This is the image
        self.template_label.setPixmap(pixmap)

        print("Loaded template image file: " + self.template_path)

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                    bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):

    	#Reference image, in grey scale
    	img = cv2.imread(self.ImgPath,cv2.IMREAD_GRAYSCALE)

    	# Create a sift object for using SIFT related functions
    	sift = cv2.xfeatures2d.SIFT_create()

    	#Get the keypoints and descriptors of the robot image
    	kp_image, desc_image = sift.detectAndCompute(img, None)

    	# Feature matching
    	index_params = dict(algorithm=0, trees=5)
    	search_params = dict()
    	flann = cv2.FlannBasedMatcher(index_params, search_params)

    	self.VidCap.set(3, 320)
    	self.VidCap.set(4, 240)

    	#Read frames of the recorded video
    	val,frame = self.VidCap.read()

    	"""Frame must be resized"""
    	wid = int(frame.shape[1]* 1/3)
    	hei = int(frame.shape[0]* 1/3)
    	dimensions = (wid,hei)

    	frame = cv2.resize(frame,dimensions,interpolation = cv2.INTER_AREA) 
    	print(frame.size)

    	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage

    	kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)

    	matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

    	good_points = []

    	for m, n in matches:
    		if m.distance < 0.6 * n.distance:
    			good_points.append(m)

    	if(len(good_points) > 10):  		
    		query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    		print(query_pts)
    		train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    		print(train_pts)
    		matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    		matches_mask = mask.ravel().tolist()		

    		# Perspective transform
    		h, w = img.shape
    		pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    		dst = cv2.perspectiveTransform(pts, matrix)

    		homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
    		pixmap = self.convert_cv_to_pixmap(frame)
    		
    		self.live_image_label.setPixmap(pixmap)

    	else:
            self.live_image_label.setPixmap(self.convert_cv_to_pixmap(frame))	


    def SLOT_toggle_camera(self):

    	""" Open the video at the specified location. """
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())

