import cv2
import numpy as np
import os
from tkinter import filedialog, Tk, Label, Button, Frame, messagebox, Toplevel
from PIL import Image, ImageTk

def match_template(image, template):
    if image.ndim == 3 and template.ndim == 2:
        # Convert image to grayscale if it's color and template is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype != np.float32:
        # Convert image to float32 for better accuracy
        image = image.astype(np.float32)

    if template.dtype != np.float32:
        # Convert template to float32 for better accuracy
        template = template.astype(np.float32)

    # Calculate result dimensions
    result_h, result_w = image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1
    if result_h <= 0 or result_w <= 0:
        raise ValueError("Template size is larger than the image size.")

    result = np.zeros((result_h, result_w), dtype=np.float32)

    # Perform template matching using cv2.TM_CCOEFF_NORMED
    template_mean = np.mean(template)
    template_std = np.std(template)
    template = (template - template_mean) / (template_std + 1e-5)
    result = cv2.filter2D(image, -1, template)

    image_sqmean = cv2.filter2D(image**2, -1, np.ones(template.shape) / np.prod(template.shape))
    image_mean = cv2.filter2D(image, -1, np.ones(template.shape) / np.prod(template.shape))
    image_var = image_sqmean - image_mean**2
    epsilon = 1e-10
    image_var = np.maximum(image_var, 0)  # Ensure non-negative variance
    image_std = np.sqrt(image_var + epsilon)
    result /= (image_std * (np.prod(template.shape) - 1) + epsilon)

    return result

class CanDetector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image not loaded correctly.")
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.equalized_image = cv2.equalizeHist(self.gray_image)

    def detect_can(self, lower_color, upper_color):
        mask = cv2.inRange(self.hsv_image, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, mask

    def apply_template_matching(self, template_folder, can_type, threshold=0.7, scale_factor=2, rect_color=(0, 255, 0)):
        matched_templates = []

        for template_name in os.listdir(template_folder):
            template_path = os.path.join(template_folder, template_name)
            template = cv2.imread(template_path, 0)
            if template is None:
                continue
            template = cv2.equalizeHist(template)
            w, h = template.shape[::-1]
            try:
                res = match_template(self.equalized_image, template)
                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):
                    expanded_w = int(w * scale_factor)
                    expanded_h = int(h * scale_factor)
                    x_start = max(0, pt[0] - (expanded_w - w) // 2)
                    y_start = max(0, pt[1] - (expanded_h - h) // 2)
                    cv2.rectangle(self.image, (x_start, y_start), (x_start + expanded_w, y_start + expanded_h), rect_color, 2)
                    cv2.putText(self.image, can_type, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rect_color, 2)
                    if template_path not in matched_templates:
                        matched_templates.append(template_path)
            except Exception as e:
                print(f"Error matching template {template_path}: {e}")
        return self.image, matched_templates

class PepsiDetector(CanDetector):
    def __init__(self, image_path, template_folder):
        super().__init__(image_path)
        self.template_folder = template_folder

    def detect(self):
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        contours, mask = self.detect_can(lower_blue, upper_blue)
        if contours:
            return self.apply_template_matching(self.template_folder, "Pepsi", rect_color=(249, 236, 6)), mask
        return self.image, [], mask

class CokeDetector(CanDetector):
    def __init__(self, image_path, template_folder):
        super().__init__(image_path)
        self.template_folder = template_folder

    def detect(self):
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        contours1, mask1 = self.detect_can(lower_red1, upper_red1)
        contours2, mask2 = self.detect_can(lower_red2, upper_red2)
        if contours1 or contours2:
            mask = cv2.bitwise_or(mask1, mask2)
            return self.apply_template_matching(self.template_folder, "Coke", rect_color=(152, 244, 20)), mask
        return self.image, [], mask




def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        status_label.config(text="Processing...")
        root.update()
        try:
            pepsi_detector = PepsiDetector(file_path, 'C://Users//USER//Downloads//Image Project Data//Pepsi_logo')
            coke_detector = CokeDetector(file_path, 'C://Users//USER//Downloads//Image Project Data//Coke_logo')
            (pepsi_result, pepsi_templates), pepsi_mask = pepsi_detector.detect()
            (coke_result, coke_templates), coke_mask = coke_detector.detect()
            display_intermediate_results(pepsi_detector, coke_detector, pepsi_mask, coke_mask, pepsi_templates, coke_templates)
            status_label.config(text="Detection completed.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            status_label.config(text="An error occurred. Try again.")

def display_intermediate_results(pepsi_detector, coke_detector, pepsi_mask, coke_mask, pepsi_templates, coke_templates):
    for widget in main_frame.winfo_children():
        if isinstance(widget, Label) and widget != status_label and widget != open_button:
            widget.destroy()

    display_image(main_frame, pepsi_mask, "Color Segmentation Pepsi", 2, 0)
    display_image(main_frame, coke_mask, "Color Segmentation Coke", 2, 1)
    display_image(main_frame, pepsi_detector.equalized_image, "Equalized Pepsi Image", 1, 0)
    display_image(main_frame, coke_detector.equalized_image, "Equalized Coke Image", 1, 1)
    display_templates(main_frame, pepsi_templates, "Matched Templates - Pepsi", 1, 2)
    display_templates(main_frame, coke_templates, "Matched Templates - Coke", 2, 2)
    display_final_result(pepsi_detector.image, coke_detector.image)

def display_image(window, image, title, row, col, colspan=1):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    desired_size = (250, 300)
    image_pil = Image.fromarray(image_rgb).resize(desired_size, Image.LANCZOS)
    image_photo = ImageTk.PhotoImage(image=image_pil)
    image_label = Label(window, image=image_photo, text=title, compound="top")
    image_label.image = image_photo
    image_label.grid(row=row, column=col, columnspan=colspan, padx=10, pady=10, sticky="nsew")

def display_image2(window, image, title, row, col, colspan=1):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    desired_size = (450, 600)
    image_pil = Image.fromarray(image_rgb).resize(desired_size, Image.LANCZOS)
    image_photo = ImageTk.PhotoImage(image=image_pil)
    image_label = Label(window, image=image_photo, text=title, compound="top")
    image_label.image = image_photo
    image_label.grid(row=row, column=col, columnspan=colspan, padx=10, pady=10, sticky="nsew")    

def display_templates(window, templates, title, row, col):
    for i, template_path in enumerate(templates):
        template_image = Image.open(template_path).resize((150, 150), Image.LANCZOS)
        template_photo = ImageTk.PhotoImage(image=template_image)
        template_label = Label(window, image=template_photo, text=title, compound="top")
        template_label.image = template_photo
        template_label.grid(row=row, column=col + i, padx=10, pady=10, sticky="nsew")

def display_final_result(pepsi_image, coke_image):
    result_window = Toplevel(root)
    result_window.title("Final Can Detection Result")
    display_image2(result_window, pepsi_image, "Final Detected Pepsi Cans", 0, 0)
    display_image2(result_window, coke_image, "Final Detected Coke Cans", 0, 1)

root = Tk()
root.title("Can Detector UI")
main_frame = Frame(root)
main_frame.pack(pady=20, padx=20)

open_button = Button(main_frame, text="Open Image", command=select_image)
open_button.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

status_label = Label(main_frame, text="Please select an image to detect cans.", relief="sunken")
status_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

root.mainloop()
