import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(0)
from tkinter import *
from chat import get_response
import numpy
import cv2
import pyautogui

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

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

    def _capture_screenshot(self, event):
        pil_file = pyautogui.screenshot()
        numpy_arr = numpy.array(pil_file)
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

if __name__ == "__main__":
    app = ChatApplication()
    app.run()