import os.path
import pickle

# GUI
from tkinter import * 
from tkinter import messagebox, simpledialog, filedialog

import numpy as np
import PIL # Pillow - Python Imaging Library
from PIL import Image, ImageDraw
import cv2 as cv

# Machine Learning models
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
        
    # prompt user for project name and class names and setup/load data
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
    
    # initialize the GUI
    def init_gui(self):
        WIDTH, HEIGHT = 500, 500
        WHITE = (255, 255, 255)
        
        self.root = Tk()
        self.root.title(f"Drawing Classifier v0.1 - {self.proj_name}")

        self.canvas = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint) # bind event of moving the mouse with left button pressed to the paint function

        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        # button frame and columns
        btn_frame = Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        # BUTTONS

        # class buttons
        class1_btn = Button(btn_frame, text=self.class1, command=lambda: self.save(1))
        class1_btn.grid(row=0, column=0, sticky=W+E) # place button in grid

        class2_btn = Button(btn_frame, text=self.class2, command=lambda: self.save(2))
        class2_btn.grid(row=0, column=1, sticky=W+E)

        class3_btn = Button(btn_frame, text=self.class3, command=lambda: self.save(3))
        class3_btn.grid(row=0, column=2, sticky=W+E)

        # brush size adjustment and clear buttons
        bm_btn = Button(btn_frame, text="Brush-", command=self.brushminus)
        bm_btn.grid(row=1, column=0, sticky=W+E)

        clear_btn = Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.grid(row=1, column=1, sticky=W+E)

        bp_btn = Button(btn_frame, text="Brush+", command=self.brushplus)
        bp_btn.grid(row=1, column=2, sticky=W+E)

        # train, save and load buttons
        train_btn = Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=2, column=0, sticky=W+E)

        save_btn = Button(btn_frame, text="Save Model", command=self.save_model)
        save_btn.grid(row=2, column=1, sticky=W+E)

        load_btn = Button(btn_frame, text="Load Model", command=self.load_model)
        load_btn.grid(row=2, column=2, sticky=W+E)

        # change and predict model and save everything buttons
        change_btn = Button(btn_frame, text="Change Model", command=self.rotate_model)
        change_btn.grid(row=3, column=0, sticky=W+E)

        predict_btn = Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=3, column=1, sticky=W+E)

        save_everything_btn = Button(btn_frame, text="Save Everything", command=self.save_everything)
        save_everything_btn.grid(row=3, column=2, sticky=W+E)

        # status label
        self.status_label = Label(btn_frame, text=f"Current Model: {type(self.clf).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4, column=1, sticky=W+E)

        # handle window closing and start the process
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes('-topmost', True)
        self.root.mainloop()

    # paint on the canvas and image
    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
        self.draw.rectangle([x1, y1, x2 + self.brush_width, y2 + self.brush_width], fill = "black", width=self.brush_width)

    # save the current drawing to the appropriate class folder
    def save(self, class_num):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.Resampling.LANCZOS) # resize image to 50x50
        
        if class_num == 1:
            img.save(f"{self.proj_name}/{self.class1}/{self.class1_counter}.png", "PNG")
            self.class1_counter += 1
        elif class_num == 2:
            img.save(f"{self.proj_name}/{self.class2}/{self.class2_counter}.png", "PNG")
            self.class2_counter += 1
        elif class_num == 3:
            img.save(f"{self.proj_name}/{self.class3}/{self.class3_counter}.png", "PNG")
            self.class3_counter += 1
        
        self.clear()

    # brush size decrease
    def brushminus(self):
        if self.brush_width > 1:
            self.brush_width -= 1
    
    # brush size increase
    def brushplus(self):
        self.brush_width += 1

    # clear the canvas
    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    # train the model with the current data
    def train_model(self):
        img_list = np.array([]) # list of all image pixels
        class_list = np.array([]) # list of all image classes

        for x in range(1, self.class1_counter):
            img = cv.imread(f"{self.proj_name}/{self.class1}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, [1])

        for x in range(1, self.class2_counter):
            img = cv.imread(f"{self.proj_name}/{self.class2}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, [2])
        
        for x in range(1, self.class3_counter):
            img = cv.imread(f"{self.proj_name}/{self.class3}/{x}.png")[:, :, 0]
            img = img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, [3])
        
        img_list = img_list.reshape(self.class1_counter - 1 + self.class2_counter - 1 + self.class3_counter - 1, 2500)
        
        self.clf.fit(img_list, class_list)

        messagebox.showinfo("Drawing Classifier", "Model successfully trained!", parent=self.root)

    # save the model
    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.clf, f)
        messagebox.showinfo("Drawing Classifier", "Model successfully saved!", parent=self.root)

    # load the model
    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as f:
            self.clf = pickle.load(f)
        messagebox.showinfo("Drawing Classifier", "Model successfully loaded!", parent=self.root)

    # rotate between different ML models
    def rotate_model(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = LogisticRegression()
        elif isinstance(self.clf, LogisticRegression):
            self.clf = DecisionTreeClassifier()
        elif isinstance(self.clf, DecisionTreeClassifier):
            self.clf = RandomForestClassifier()
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf = GaussianNB()
        elif isinstance(self.clf, GaussianNB):
            self.clf = LinearSVC()
        
        self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")

    # predict the class of the current drawing
    def predict(self):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.Resampling.LANCZOS)
        img.save("predictshape.png", "PNG")

        img = cv.imread("predictshape.png")[:, :, 0]
        img = img.reshape(2500)
        prediction = self.clf.predict([img])

        if prediction[0] == 1:
            messagebox.showinfo("Drawing Classifier", f"The drawing is probably a {self.class1}", parent=self.root)
        elif prediction[0] == 2:
            messagebox.showinfo("Drawing Classifier", f"The drawing is probably a {self.class2}", parent=self.root)
        elif prediction[0] == 3:
            messagebox.showinfo("Drawing Classifier", f"The drawing is probably a {self.class3}", parent=self.root)

    # save all data
    def save_everything(self):
        data = {
            'c1': self.class1,
            'c2': self.class2,
            'c3': self.class3,
            'c1c': self.class1_counter,
            'c2c': self.class2_counter,
            'c3c': self.class3_counter,
            'clf': self.clf,
            'pname': self.proj_name
        }
        with open(f"{self.proj_name}/{self.proj_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)

        messagebox.showinfo("Drawing Classifier", "Project successfully saved!", parent=self.root)    

    # handle window closing event
    def on_closing(self):
        answer = messagebox.askyesnocancel("Quit?", "Do you want to save your work?", parent=self.root)
        if answer is not None:    
            if answer:
                self.save_everything()
            self.root.destroy()
            exit()

DrawingClassifier()
