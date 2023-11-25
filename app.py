# needed to fix the scaling issue on windows due to pyautogui
#import ctypes
#ctypes.windll.shcore.SetProcessDpiAwareness(0)

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

# used for speech recognition
import speech_recognition as sr
from vosk import Model, KaldiRecognizer, SetLogLevel
import pyaudio

# used for llm
import openai
import json
import argparse
import re
import os

from deevee_wrapper import *
from history import *

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)

class colors:  # You may need to change color settings
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

dw = DeeveeWrapper()

class MainApplication:
    def __init__(self):
        #dw = DeeveeWrapper()

        self._initialize_chatgpt()

        # Initialize Tkinter
        #self.window = Tk()
        #self._setup_main_window()        
        
        # Initialize the history object
        self.dv_history = History("history")
        
        #Loads the model for performing inference
        self.model = tf.saved_model.load("models/od_stage2")

        with open(self.args.prompt, "r") as f:
            prompt = f.read()

        self._ask(prompt)

        print("Welcome to dv! Type 'help' for a list of commands.")
        while True:
            question = input(colors.YELLOW + "Deevee> " + colors.ENDC)

            if question == "!quit" or question == "!exit":
                break

            if question == "!clear":
                os.system("cls")
                continue

            # Update the desktop state
            dw.get_desktop_state()
            # modify the question to include the desktop state
            modified_question = "This is the desktop state: \n\n" + dw.print_desktop_state() + "\n\n" + question

            response = self._ask(modified_question)

            print(f"\n{response}\n")

            code = self._extract_python_code(response)
            if code is not None:
                print("Please wait while I execute the code...")
                exec(self._extract_python_code(response))
                print("Done!\n")

        # Initialize the vosk model and recognizer (Method 1)
        #self.microphone = sr.Microphone()
        #SetLogLevel(0)
        #speech_model = Model("models/vosk-model-small-en-us-0.15")
        #self.recognizer = KaldiRecognizer(speech_model, 16000)

        #self.mic = pyaudio.PyAudio()
        #self.stream = self.mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192) 
        #self.stream.start_stream()

        #while True:
        #    data = self.stream.read(4096)
        #    if self.recognizer.AcceptWaveform(data):
        #        text = self.recognizer.Result()
        #        print(f"' {text[14:-3]} '")

        #        if text[14:-3] == "skip":
        #            self._on_skip_pressed(None)        

        # Initialize the recognizer (Method 2)
        #self.recognizer = sr.Recognizer()
        #self.microphone = sr.Microphone()
        #with self.microphone as source:
        #    self.recognizer.adjust_for_ambient_noise(source)

        # Start listening in the background (Method 2)
        #self.stop_listening = self.recognizer.listen_in_background(self.microphone, self._listen_callback)

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
                filtered_predictions["detection_boxes"].append(predictions["detection_boxes"][0][i])
                filtered_predictions["detection_classes"].append(predictions["detection_classes"][0][i])

        print(filtered_predictions)

        # if filtered_predictions is not empty, calculate the center of the bounding box of the first element
        # and move the mouse to that location and click
        if len(filtered_predictions["detection_scores"]) > 0:
            # calculate the center of the bounding box
            x1 = filtered_predictions["detection_boxes"][0][1]
            y1 = filtered_predictions["detection_boxes"][0][0]
            x2 = filtered_predictions["detection_boxes"][0][3]
            y2 = filtered_predictions["detection_boxes"][0][2]

            x = (x1 + x2) / 2
            y = (y1 + y2) / 2

            dw.press_lmb(x, y)

    def _listen_callback(self, recognizer, audio):
        try:
            query = recognizer.recognize_whisper(audio, "tiny.en")
            print(f"User said: {query}\n")
        except sr.UnknownValueError:
            print("SR could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from SR service; {e}")

    def _initialize_chatgpt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", type=str, default="prompts/dv_basic.txt")
        parser.add_argument("--sysprompt", type=str, default="system_prompts/deevee_basic.txt")
        self.args = parser.parse_args()

        # Initialize OpenAI API
        with open("config.json", "r") as f:
            config = json.load(f)
        print("Initializing ChatGPT...")
        openai.api_key = config["OPENAI_API_KEY"]

        with open(self.args.sysprompt, "r") as f:
            sysprompt = f.read()

        self.chat_history = [
            {
                "role": "system",
                "content": sysprompt
            },
            {
                "role": "user",
                "content": "move the mouse to the top left of the screen"
            },
            {
                "role": "assistant",
                "content": """```python
                    dw.move_mouse(0,0)
                    ```

                    This code uses the `move_mouse()` function to move the mouse to the center of the screen."""
            }
        ]

        print("ChatGPT initialized.")

    def _ask(self, prompt):
        self.chat_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-1106:personal::8N16lm9V",
            messages=self.chat_history,
            temperature=0
        )
        self.chat_history.append(
            {
                "role": "assistant",
                "content": completion.choices[0].message.content,
            }
        )

        self.desktop_state_tag = 'desktop_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dv_history.store_user_prompt(prompt, completion.choices[0].message.content, self.desktop_state_tag)
        return self.chat_history[-1]["content"]

    def _extract_python_code(self, content):
        code_blocks = code_block_regex.findall(content)
        if code_blocks:
            full_code = "\n".join(code_blocks)

            if full_code.startswith("python"):
                full_code = full_code[7:]

            return full_code
        else:
            return None
    
if __name__ == "__main__":
    app = MainApplication()
    app.run()