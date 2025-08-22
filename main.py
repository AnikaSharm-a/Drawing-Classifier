import os.path
import pickle

# GUI
from tkinter import * 
from tkinter import messagebox, simpledialog

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
        msg = Tk()
        msg.withdraw()

        self.proj_name = simpledialog.askstring("Project Name", "Please enter your project name down below!", parent=msg)

        # if directory exists, open the pickle file and extract info
        if os.path.exists(self.proj_name):
            with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            
            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class3 = data['c3']
            
            self.class1_counter = data['c1c']
            self.class2_counter = data['c2c']
            self.class3_counter = data['c3c']
            
            self.clf = data['clf']
            self.proj_name = data['pname']
        
        # otherwise, create the new directory with all the data
        else:
            self.class1 = simpledialog.askstring("Class 1", "What is the first class called?", parent=msg)
            self.class2 = simpledialog.askstring("Class 2", "What is the second class called?", parent=msg)
            self.class3 = simpledialog.askstring("Class 3", "What is the third class called?", parent=msg)

            self.class1_counter, self.class2_counter, self.class3_counter = 1, 1, 1

            self.clf = LinearSVC() # by default, use Linear Support Vector Classifier

            # project directory structure
            os.mkdir(self.proj_name)
            os.chdir(self.proj_name)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir("..")
    
    def init_gui(self):
        pass
