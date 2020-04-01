"""
author: RohitMidha23
description: A simple code to convert annotations from VOC format to COCO Format.
"""


import os
import xml.etree.ElementTree as ET
import xmltodict
import json
from xml.dom import minidom
from collections import OrderedDict


def generateVOC2Json(rootDir, xmlFiles):
    attrDict = dict()
    # List your classes here
    attrDict["categories"] = [
        {"supercategory": "none", "id": 1, "name": "1"},
        {"supercategory": "none", "id": 2, "name": "auto"},
        {"supercategory": "none", "id": 3, "name": "bus"},
        {"supercategory": "none", "id": 4, "name": "truck"},
        {"supercategory": "none", "id": 5, "name": "car"},
    ]
    images = list()
    annotations = list()
    for root, dirs, files in os.walk(rootDir):
        image_id = 0
        for file in xmlFiles:
            image_id = image_id + 1
            if file in files:
                annotation_path = os.path.abspath(os.path.join(root, file))
                image = dict()
                doc = xmltodict.parse(open(annotation_path).read())
                image["file_name"] = str(doc["annotation"]["filename"])
                image["height"] = int(doc["annotation"]["size"]["height"])
                image["width"] = int(doc["annotation"]["size"]["width"])
                image["id"] = image_id
                print("[INFO]File Name: {} and image_id {}".format(file, image_id))
                images.append(image)
                id1 = 1
                if "object" in doc["annotation"]:
                    for obj in doc["annotation"]["object"]:
                        for value in attrDict["categories"]:
                            annotation = dict()

                            if str(obj["name"]) == value["name"]:
                                annotation["iscrowd"] = 0
                                annotation["image_id"] = image_id
                                x1 = int(obj["bndbox"]["xmin"]) - 1
                                y1 = int(obj["bndbox"]["ymin"]) - 1
                                x2 = int(obj["bndbox"]["xmax"]) - x1
                                y2 = int(obj["bndbox"]["ymax"]) - y1
                                annotation["bbox"] = [x1, y1, x2, y2]
                                annotation["area"] = float(x2 * y2)
                                annotation["category_id"] = value["id"]
                                annotation["ignore"] = 0
                                annotation["id"] = id1
                                annotation["segmentation"] = [
                                    [
                                        x1,
                                        y1,
                                        x1,
                                        (y1 + y2),
                                        (x1 + x2),
                                        (y1 + y2),
                                        (x1 + x2),
                                        y1,
                                    ]
                                ]
                                id1 += 1

                                annotations.append(annotation)

                else:
                    print("[INFO]File: {} doesn't have any object".format(file))

            else:
                print("[INFO]File: {} not found".format(file))

    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    jsonString = json.dumps(attrDict)
    with open("train.json", "w") as f:
        f.write(jsonString)


trainFile = "train.txt"
trainXMLFiles = list()
with open(trainFile, "rb") as f:
    for line in f:
        fileName = line.strip()
        fileName = str(fileName, "utf-8")
        fileName = fileName.strip(".jpg")
        print("[INFO]Filename: ", fileName)
        trainXMLFiles.append(fileName + ".xml")


rootDir = "train_xmls/"
generateVOC2Json(rootDir, trainXMLFiles)
