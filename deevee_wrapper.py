import pyautogui

import numpy as np
import io
import cv2
import tensorflow as tf # ver = 2.13
from PIL import Image
from array import array

regions_dict = {
    1: "DESKTOP",
    2: "TASKBAR",
    3: "WINDOW"
}

objects_dict = {
    1: "SKIP_AD_BUTTON",
}

class DeeveeWrapper:
    def __init__(self):
        #Loads the model for performing inference
        self.model = tf.saved_model.load("model")
        self.model_stage1 = tf.saved_model.load("models/od_stage1")

    def click(self):
        pyautogui.click()

    def move_mouse(self, x, y):
        pyautogui.moveTo(x * pyautogui.size()[0], y * pyautogui.size()[1])

    def get_object_center(self, bbox):
        return (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
    
    def get_object(self, object_name):
        for i in range(len(self.desktop_state["detection_classes"])):
            if self.desktop_state["detection_classes"][i] == object_name:
                return self.desktop_state["detection_boxes"][i]
        return None

    def get_desktop_state(self):
        pil_file = pyautogui.screenshot()
        numpy_arr = np.array(pil_file)
        current_image = cv2.cvtColor(numpy_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite('screenshot.png', current_image)

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
        predictions = self.model_stage1(input_tensor)

        # Filter predictions based on detection_scores above 0.8
        self.desktop_state = {
            "detection_scores": [],
            "detection_boxes": [],
            "detection_classes": []
        }

        for i in range(len(predictions["detection_scores"][0])):
            if predictions["detection_scores"][0][i] > 0.50: #.20 for SKIP_AD_BUTTON
                self.desktop_state["detection_scores"].append(predictions["detection_scores"][0][i].numpy())
                self.desktop_state["detection_boxes"].append(predictions["detection_boxes"][0][i].numpy())
                #self.desktop_state["detection_classes"].append(objects_dict[int(predictions["detection_classes"][0][i].numpy())])
                self.desktop_state["detection_classes"].append(regions_dict[int(predictions["detection_classes"][0][i].numpy())])
        
        return self.desktop_state
