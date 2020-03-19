"""
author: RohitMidha23
description: A simple code to convert annotations from VOC format to YOLO format for object detection.
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# Enter your classes here
classes = ["1", "2", "3", "4"] 


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(in_path, out_path):
    in_file = open(in_path)
    out_file = open(out_path, "w")
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text),
        )
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")


in_folder = input("Enter Label Folder: ")
out_folder = input("Enter Output Folder: ")

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

for image_id_full in os.listdir(in_folder):
    image_id = image_id_full.split(".xml")[0]
    in_path = os.path.join(in_folder, image_id_full)
    out = image_id + ".txt"
    out_path = os.path.join(out_folder, out)
    convert_annotation(in_path, out_path)
