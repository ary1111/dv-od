import pyautogui

import numpy as np
import io
import cv2
import tensorflow as tf # ver = 2.13
from PIL import Image
from array import array

from history import *
import easyocr

import datetime
import random
import string

regions_dict = {
    1: "DESKTOP",
    2: "TASKBAR",
    3: "WINDOW"
}

objects_dict = {
    1: "SKIP_AD_BUTTON",
    2: "MINIMIZE_BUTTON",
    3: "CLOSE_BUTTON",
    4: "CARD",
    5: "TEXT_BUTTON"
}

class DeeveeWrapper:
    def __init__(self):
        # Loads the models for performing inference
        self.model_stage1 = tf.saved_model.load("models/od_stage1")
        self.model_stage2 = tf.saved_model.load("models/od_stage2")

        # Initialize the easyocr reader for the specified language
        self.ocr = easyocr.Reader(['en'])

        # Initialize the history object
        self.dv_history = History("history")

    def click(self):
        pyautogui.click()

    def move_mouse(self, x, y):
        pyautogui.moveTo(x * pyautogui.size()[0], y * pyautogui.size()[1])

    def get_object_center(self, bbox):
        return (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
    
    def get_object(self, object_name):
        for region in self.desktop_state["regions"]:
            for i in range(len(region["children"])):
                if region["children"][i]["class"] == object_name:
                    return region["children"][i]["box"]
        return None

    def get_desktop_state(self):
        import datetime

        pil_file = pyautogui.screenshot()
        numpy_arr = np.array(pil_file)        
        current_image = cv2.cvtColor(numpy_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite('screenshot.png', current_image)

        self.desktop_state_tag = 'desktop_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dv_history.store_desktop_image(pil_file, self.desktop_state_tag)

        orig_image = Image.open('screenshot.png')

        # Preprocess the image and convert to tensor)
        input_tensor = self.preprocess_image(orig_image)

        # Run Stage 1 inference
        s1predictions = self.model_stage1(input_tensor)

        self.desktop_state = {
            "regions":[]
        }

        for i in range(len(s1predictions["detection_scores"][0])):
            if s1predictions["detection_scores"][0][i] > 0.50: #.20 for SKIP_AD_BUTTON
                region = {
                    "class": regions_dict[int(s1predictions["detection_classes"][0][i].numpy())],
                    "score": s1predictions["detection_scores"][0][i].numpy(),
                    "box": s1predictions["detection_boxes"][0][i].numpy(),
                    "children": []
                }
                self.desktop_state["regions"].append(region)

        self.dv_history.store_region_info(self.desktop_state, self.desktop_state_tag)
        
        # Run Stage 2 inference
        for region in self.desktop_state["regions"]:
            if region["class"] == "WINDOW":

                cropped_image = orig_image.crop((
                    int(region["box"][1] * pyautogui.size()[0]), #left
                    int(region["box"][0] * pyautogui.size()[1]), #upper
                    int(region["box"][3] * pyautogui.size()[0]), #right
                    int(region["box"][2] * pyautogui.size()[1])  #lower
                ))
            
                # Preprocess the image and convert to tensor)
                cropped_tensor = self.preprocess_image(cropped_image)

                #calculate the region height and width (normalized)
                region_height = region["box"][2] - region["box"][0]
                region_width = region["box"][3] - region["box"][1]

                predictions = self.model_stage2(cropped_tensor)
                for i in range(len(predictions["detection_scores"][0])):
                    if predictions["detection_scores"][0][i] > 0.20:
                        # convert detection_boxes tensor to numpy array
                        detection_boxes = predictions["detection_boxes"][0][i].numpy()

                        #scale the detection_boxes to the original image size (normalized in desktop space)
                        detection_boxes[0] = detection_boxes[0] * region_height + region["box"][0]
                        detection_boxes[1] = detection_boxes[1] * region_width + region["box"][1]
                        detection_boxes[2] = detection_boxes[2] * region_height + region["box"][0]
                        detection_boxes[3] = detection_boxes[3] * region_width + region["box"][1]

                        region["children"].append({
                            "class": objects_dict[int(predictions["detection_classes"][0][i].numpy())],
                            "score": predictions["detection_scores"][0][i].numpy(),
                            "box": detection_boxes
                        })

                # OCR performed on CARD objects
                #for i in range(len(region["children"])):
                #    if region["children"][i]["class"] == "CARD":
                #        cropped_image = orig_image.crop((
                #            int(region["children"][i]["box"][1] * pyautogui.size()[0]), #left
                #            int(region["children"][i]["box"][0] * pyautogui.size()[1]), #upper
                #            int(region["children"][i]["box"][3] * pyautogui.size()[0]), #right
                #            int(region["children"][i]["box"][2] * pyautogui.size()[1])  #lower
                #        ))
                #        ocr_predictions = self.perform_ocr(cropped_image)
                #        print(ocr_predictions)

                ocr_predictions = self.perform_ocr(cropped_image)
                print(ocr_predictions)

                # Store any text predictions that overlap with the CARD objects
                for i in range(len(region["children"])):
                    if region["children"][i]["class"] == "CARD":
                        print("Region: %s" % region["children"][i]["box"])                        
                        # OCR predictions are in the form of [[text, [x1, y1], [x2, y2], [x3, y3], [x4, y4]]]
                        for j in range(len(ocr_predictions)):
                            #print("%s: %s" % (ocr_predictions[j][1], ocr_predictions[j][0]))
                            center_x = float(ocr_predictions[j][0][0][0] + ocr_predictions[j][0][1][0]) / 2 #pixels, cropped
                            center_x = center_x / (region_width*pyautogui.size()[0]) + region["box"][1]     #normalized, desktop
                            center_y = float(ocr_predictions[j][0][0][1] + ocr_predictions[j][0][2][1]) / 2 #pixels, cropped
                            center_y = center_y / (region_height*pyautogui.size()[1]) + region["box"][0]    #normalized, desktop                            
                            ocr_center = [center_x, center_y]                                           

                            #print(ocr_center)
                            # Check if ocr_center is within the CARD object
                            if ocr_center[0] > region["children"][i]["box"][1] and ocr_center[0] < region["children"][i]["box"][3]:
                                if ocr_center[1] > region["children"][i]["box"][0] and ocr_center[1] < region["children"][i]["box"][2]:
                                    region["children"][i]["text"] = ocr_predictions[j][0]
                                    print("%s: %s" % (ocr_predictions[j][1],ocr_predictions[j][0]))        
                
                self.dv_history.store_stage2_info(region, region_height*pyautogui.size()[1], region_width*pyautogui.size()[0], self.desktop_state_tag)
                self.dv_history.store_cropped_image(cropped_image, self.desktop_state_tag)
        
        return self.desktop_state
    
    def preprocess_image(self, image):
        image = image.resize((640, 640))
        image = image.convert('RGB')
        image = np.array(image)
        image = image.astype(np.uint8)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        image_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        image_tensor = image_tensor[tf.newaxis, ...]

        return image_tensor
    
    def perform_ocr(self, image):
        #cropped_image = cropped_image.resize((640, 640))
        cropped_image = image.convert('RGB')
        image_array = np.array(cropped_image)
        image_array = image_array.astype(np.uint8)
                
        #return self.ocr.recognize([image_array])
        return self.ocr.readtext(image_array)