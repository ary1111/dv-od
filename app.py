# needed to fix the scaling issue on windows due to pyautogui
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(0)

# used for the GUI
from tkinter import *

# used for screenshot
import pyautogui

# used for plotting
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# used for machine learning
import numpy as np
import io
import cv2
import tensorflow as tf # ver = 2.13
from PIL import Image

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class MainApplication:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

        #Loads the model for performing inference
        self.model = tf.saved_model.load("model")

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("dv - Virtual Assistant and Interface")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=800, height=550, bg=BG_COLOR)

        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # Creating Figure.
        self.fig = Figure(figsize = (18,10), dpi = 100)

        # Creating Canvas
        self.canv = FigureCanvasTkAgg(self.fig, master = self.window)
        self.canv.get_tk_widget().place(relx=0.05, rely=0.1, relwidth=0.9, relheight=0.7)

        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # skip button
        skip_button = Button(bottom_label, text="Skip", font=FONT_BOLD, width=20, bg=BG_GRAY,
                                   command=lambda: self._on_skip_pressed(None))
        skip_button.place(relx=0.39, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_skip_pressed(self, event):
        self._capture_screenshot(None)
        self._perform_inference(None)

    def _capture_screenshot(self, event):
        pil_file = pyautogui.screenshot()
        numpy_arr = np.array(pil_file)
        self.current_image = cv2.cvtColor(numpy_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite('screenshot.png', self.current_image)

        # Calculate the aspect ratio of the image
        aspect_ratio = self.current_image.shape[1] / self.current_image.shape[0]

        # Calculate the width and height of the plot
        plot_width = 0.9 * self.window.winfo_width()
        plot_height = plot_width / aspect_ratio

        # Clear the figure
        self.fig.clf()

        # Display the image in the figure
        a = self.fig.add_subplot(111)
        a.imshow(self.current_image, aspect="auto")
        a.set_title("Current Image")

        # Set the size of the plot
        self.fig.set_size_inches(plot_width/100, plot_height/100)

        self.canv.draw()

        # Use place method to maintain the placement of the canvas
        self.canv.get_tk_widget().place(relx=0.05, rely=0.1, relwidth=0.9, relheight=0.7)
    
    def _perform_inference(self, event):
        image = Image.open('screenshot.png')
        # Preprocess the image (resize and convert to NumPy array)
    
        image = image.resize((640, 640))  # Resize to your desired dimensions
        image = image.convert('RGB')
        image = np.array(image)
        image = image.astype(np.uint8)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        predictions = self.model(input_tensor)

        # Filter predictions based on detection_scores above 0.8
        filtered_predictions = {
            "detection_scores": [],
            "detection_boxes": [],
            "detection_classes": []
        }

        for i in range(len(predictions["detection_scores"][0])):
            if predictions["detection_scores"][0][i] > 0.25:
                print(predictions["detection_scores"][0][i])
                filtered_predictions["detection_scores"].append(predictions["detection_scores"][0][i])
                filtered_predictions["detection_boxes"].append(predictions["detection_boxes"][0][i].numpy)
                filtered_predictions["detection_classes"].append(predictions["detection_classes"][0][i].numpy)

        print(filtered_predictions)

if __name__ == "__main__":
    app = MainApplication()
    app.run()