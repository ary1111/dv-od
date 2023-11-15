import os
from PIL import Image
import pyautogui

class History:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.crop_index = 0

    def reset_crop_index(self):
        self.crop_index = 0

    def store_desktop_image(self, image, filename):
        image.save(f"{self.output_dir}/{filename}.png", "PNG")

    def store_cropped_image(self, image, filename):
        image.save(f"{self.output_dir}/{filename}_{self.crop_index}.png", "PNG")
        self.crop_index += 1

    def store_region_info(self, desktop_state, filename):
        with open(f"{self.output_dir}/{filename}.xml", "w") as f:
            f.write("<annotation>\n")
            f.write(f"  <filename>{filename}.png</filename>\n")
            f.write(f"  <size>\n")
            f.write(f"    <width>{pyautogui.size()[0]}</width>\n")
            f.write(f"    <height>{pyautogui.size()[1]}</height>\n")
            f.write(f"  </size>\n")
            for region in desktop_state["regions"]:
                f.write("  <object>\n")
                f.write(f"    <name>{region['class']}</name>\n")
                f.write("    <bndbox>\n")
                f.write(f"      <xmin>{int(region['box'][1]*pyautogui.size()[0])}</xmin>\n")
                f.write(f"      <ymin>{int(region['box'][0]*pyautogui.size()[1])}</ymin>\n")
                f.write(f"      <xmax>{int(region['box'][3]*pyautogui.size()[0])}</xmax>\n")
                f.write(f"      <ymax>{int(region['box'][2]*pyautogui.size()[1])}</ymax>\n")
                f.write("    </bndbox>\n")
                f.write("  </object>\n")
            f.write("</annotation>\n")

    def store_stage2_info(self, region_state, height, width, filename):
        with open(f"{self.output_dir}/{filename}_{self.crop_index}.xml", "w") as f:
            f.write("<annotation>\n")
            f.write(f"  <filename>{filename}_{self.crop_index}.png</filename>\n")
            f.write(f"  <size>\n")
            f.write(f"    <width>{int(width)}</width>\n")
            f.write(f"    <height>{int(height)}</height>\n")
            f.write(f"  </size>\n")
            for region in region_state["children"]:
                f.write("  <object>\n")
                f.write(f"    <name>{region['class']}</name>\n")
                f.write("    <bndbox>\n")
                f.write(f"      <xmin>{int((region['box'][1]-region_state['box'][1])*pyautogui.size()[0])}</xmin>\n")
                f.write(f"      <ymin>{int((region['box'][0]-region_state['box'][0])*pyautogui.size()[1])}</ymin>\n")
                f.write(f"      <xmax>{int((region['box'][3]-region_state['box'][1])*pyautogui.size()[0])}</xmax>\n")
                f.write(f"      <ymax>{int((region['box'][2]-region_state['box'][0])*pyautogui.size()[1])}</ymax>\n")
                f.write("    </bndbox>\n")
                f.write("  </object>\n")
                for child in region["children"]:
                    f.write("  <object>\n")
                    f.write(f"    <name>{child['class']}</name>\n")
                    f.write("    <bndbox>\n")
                    f.write(f"      <xmin>{int((child['box'][1]-region_state['box'][1])*pyautogui.size()[0])}</xmin>\n")
                    f.write(f"      <ymin>{int((child['box'][0]-region_state['box'][0])*pyautogui.size()[1])}</ymin>\n")
                    f.write(f"      <xmax>{int((child['box'][3]-region_state['box'][1])*pyautogui.size()[0])}</xmax>\n")
                    f.write(f"      <ymax>{int((child['box'][2]-region_state['box'][0])*pyautogui.size()[1])}</ymax>\n")
                    f.write("    </bndbox>\n")
                    f.write("  </object>\n")
            f.write("</annotation>\n")

    def store_user_prompt(self, prompt, response, filename):
        with open(f"{self.output_dir}/{filename}.txt", "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {response}\n")