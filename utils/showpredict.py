import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torchvision.models.segmentation

class ImagePredictorApp:
    def __init__(self, root, model, class_to_color):
        self.root = root
        self.model = model
        self.class_to_color = class_to_color

        self.image_label = None
        self.pred_label = None

        self.setup_ui()

    def setup_ui(self):
        self.root.title("Image Predictor")

        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.import_button = tk.Button(self.frame, text="Import Image", command=self.import_image)
        self.import_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(self.frame, text="Clear", command=self.clear_image)
        self.clear_button.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.frame, width=800, height=400)
        self.canvas.pack()

    def import_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)
            self.predict_and_display(file_path)

    def display_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((128, 128), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(image)

        if self.image_label is None:
            self.image_label = self.canvas.create_image(200, 200, image=self.img)
        else:
            self.canvas.itemconfig(self.image_label, image=self.img)

    def clear_image(self):
        self.canvas.delete("all")
        self.image_label = None
        self.pred_label = None

    def predict_and_display(self, file_path):
        image = Image.open(file_path)
        image = image.resize((128, 128), Image.LANCZOS)
        image_np = np.array(image)
        image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

        self.model.eval()
        with torch.no_grad():
            pred_mask = self.model(image_tensor)['out'].squeeze().cpu().numpy()
        
        pred_mask = np.argmax(pred_mask, axis=0)
        label_rgb = self.class_idx_to_rgb(pred_mask)

        label_rgb = Image.fromarray(label_rgb)
        label_rgb = label_rgb.resize((128, 128), Image.LANCZOS)
        self.pred_img = ImageTk.PhotoImage(label_rgb)

        if self.pred_label is None:
            self.pred_label = self.canvas.create_image(600, 200, image=self.pred_img)
        else:
            self.canvas.itemconfig(self.pred_label, image=self.pred_img)

    def class_idx_to_rgb(self, label):
        rgb_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for class_idx, color in self.class_to_color.items():
            mask = label == class_idx
            rgb_label[mask] = np.array(color)
        return rgb_label

if __name__ == "__main__":
    # Define the palette and class to color mapping
    palette = [
        (0, 0, 0), (230, 25, 75), (60, 180, 75), (255, 225, 25), 
        (0, 130, 200), (145, 30, 180), (70, 240, 240), (240, 50, 230), 
        (210, 245, 60), (230, 25, 75), (0, 128, 128), (170, 110, 40), 
        (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), 
        (250, 190, 190), (0, 0, 128), (128, 128, 128),
    ]
    class_to_color = {idx: color for idx, color in enumerate(palette)}

    # Load the saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=19).to(device)
    checkpoint = torch.load('resnet_bestmodel.pt', map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])

    root = tk.Tk()
    app = ImagePredictorApp(root, model, class_to_color)
    root.mainloop()
