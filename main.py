import os.path
import pickle

# GUI
from tkinter import * 
import tkinter.messagebox

import numpy as np
import PIL # Pillow - Python Imaging Library
import cv2 as cv

# Machine Learning models
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class DrawingClassifier:
    
    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None # the three drawing classes
        self.class1_counter, self.class2_counter, self.class3_counter = None, None, None # number of images stored for each class
        self.clf = None # the classifier (ML model for prediction and training)
        self.proj_name = None
        self.root = None # main tkinter window
        self.image1 = None # taking canvas and turning into image for drawing and passing into model

        self.status_label = None
        self.canvas = None
        self.draw = None # PIL ImageDraw object for drawing on the image

        self.brush_width = 15

        self.classes_prompt()
        self.init_gui()
        
    def classes_prompt(self):
        pass

    def init_gui(self):
        pass
