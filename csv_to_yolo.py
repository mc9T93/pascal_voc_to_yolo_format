import pandas as pd
import numpy as np
from PIL import Image
import os

image_folder = "source/datasets/images/train/"

df = pd.read_csv("cardatasettrain.csv")
df = df.drop("Unnamed: 0", axis=1)
df_names = df["image"]
df_names = df_names.str.split('.', expand=True)
df_names = df_names.set_axis(['num', 'JPG'], axis='columns')
df_names = df_names.drop("JPG", axis=1)
df = df.drop("image", axis=1)
"""print(df)
print(df_names)"""

shapes = []

for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        img = Image.open(image_path)
        width, height = img.size
        shapes.append((width, height))

df['width'], df['height'] = zip(*shapes)
yolo_df = df.copy()
yolo_df["x_center"] = ((yolo_df["x1"] + yolo_df["x2"]) / 2) / yolo_df["width"]
yolo_df["x_center"] = [format(i, ".6f") for i in yolo_df["x_center"]]
yolo_df["y_center"] = ((yolo_df["y1"] + yolo_df["y2"]) / 2) / yolo_df["height"]
yolo_df["y_center"] = [format(i, ".6f") for i in yolo_df["y_center"]]
yolo_df["width"] = abs(yolo_df["x1"] - yolo_df["x2"]) / yolo_df["width"]
yolo_df["width"] = [format(i, ".6f") for i in yolo_df["width"]]
yolo_df["height"] = abs(yolo_df["y1"] - yolo_df["y2"])/ yolo_df["height"]
yolo_df["height"] = [format(i, ".6f") for i in yolo_df["height"]]
yolo_df.drop(["x1", "y1", "x2", "y2"], axis=1, inplace=True)
yolo_df = yolo_df[["Class", "x_center", "y_center", "width", "height"]]
yolo_df["Class"] = [i - 1 for i in yolo_df["Class"]]
convert_type = {

    "width": float,
    "height": float,
    "x_center": float,
    "y_center": float
}

yolo_df = yolo_df.astype(convert_type)
print(yolo_df.info(), yolo_df)

i = 0
for row in yolo_df.values:
    file_title = "source/datasets/labels/train/" + df_names["num"][i] + ".txt"
    row.tofile(file_title, sep=" ", format="%s")
    i += 1
