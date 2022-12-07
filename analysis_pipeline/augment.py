import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

import xml.dom.minidom
import xml.etree.ElementTree as ET

ia.seed(1)

seq = iaa.Sequential([
    iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0.0, 3.0))),
    iaa.Sometimes(0.1, iaa.MotionBlur(k=15)),
    iaa.Sometimes(0.1, iaa.imgcorruptlike.Saturate(severity=2)),
    iaa.Sometimes(0.1, iaa.imgcorruptlike.Saturate(severity=4)),
	iaa.Sometimes(0.1, iaa.imgcorruptlike.Brightness(severity=1)),
    iaa.Sometimes(0.1, iaa.imgcorruptlike.Brightness(severity=1)),
	iaa.Sometimes(0.1, iaa.imgcorruptlike.Contrast(severity=1)),
    iaa.Sometimes(0.1, iaa.imgcorruptlike.Contrast(severity=1)),
    iaa.Sometimes(0.1, iaa.ChangeColorTemperature((1100, 10000))),
    iaa.Sometimes(0.5, iaa.Fliplr(1)),
    iaa.Sometimes(0.5, iaa.Flipud(1)),
    iaa.Sometimes(0.25, iaa.Rot90((1, 3))),
    iaa.Sometimes(0.1, iaa.SaltAndPepper(0.1, per_channel=True)),
    iaa.Sometimes(0.1, iaa.imgcorruptlike.DefocusBlur(severity=1)),
    iaa.Sometimes(0.1, iaa.pillike.EnhanceSharpness()),
    iaa.Sometimes(0.1, iaa.imgcorruptlike.JpegCompression(severity=2)),
], random_order=True) # apply augmenters in random order

rot1 = iaa.Sequential([])


def read_xml(xml_file: str):

	tree = ET.parse(xml_file)
	root = tree.getroot()

	bounding_boxes = []

	for boxes in root.iter('object'):

		filename = root.find('filename').text

		ymin, xmin, ymax, xmax, label = None, None, None, None, None

		ymin = int(boxes.find("bndbox/ymin").text)
		xmin = int(boxes.find("bndbox/xmin").text)
		ymax = int(boxes.find("bndbox/ymax").text)
		xmax = int(boxes.find("bndbox/xmax").text)
		label = boxes.find("name").text

		bounding_boxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=label))

	return bounding_boxes

def save_pascal_xml(path,filename,image,bbs):

	root = ET.Element("annotation")

	ET.SubElement(root, "folder").text = "output"
	ET.SubElement(root, "filename").text = filename.replace("xml","jpg")
	ET.SubElement(root, "path").text = path+"/"+filename.replace("xml","jpg")

	source = ET.SubElement(root, "source")
	ET.SubElement(source, "database").text = "Unknown"

	size = ET.SubElement(root, "size")
	ET.SubElement(size, "width").text = str(image.shape[0])
	ET.SubElement(size, "height").text = str(image.shape[1])
	ET.SubElement(size, "depth").text = "3" 

	ET.SubElement(root, "segmented").text = "0"

	for bb in bbs:
		obj = ET.SubElement(root, "object")
		ET.SubElement(obj, "name").text = str(bb.label)
		ET.SubElement(obj, "pose").text = "Unspecified"
		ET.SubElement(obj, "truncated").text = "0"
		ET.SubElement(obj, "difficult").text = "0"

		bndbox = ET.SubElement(obj, "bndbox")
		ET.SubElement(bndbox, "xmin").text = str(bb.x1)
		ET.SubElement(bndbox, "ymin").text = str(bb.y1)
		ET.SubElement(bndbox, "xmax").text = str(bb.x2)
		ET.SubElement(bndbox, "ymax").text = str(bb.y2)

	tree = ET.ElementTree(root)

	tree.write(path+"/"+filename)

def convert_images(path, output_num, iter=10, padColor=0):

	output = []

	path = os.path.join(path)

	for i in range(iter):
		for img in tqdm(os.listdir(path)):

			if(".jpg" not in img):
				continue

			img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)

			bbs = read_xml(str(path+"/"+img).replace("jpg","xml"))

			image_aug, bbs_aug = seq(image=img_array, bounding_boxes=bbs)

			cv2.imwrite("output/image"+str(output_num)+".jpg",image_aug)

			save_pascal_xml("output","image"+str(output_num)+".xml",image_aug,bbs_aug)

			output_num += 1

	return output_num

output_num = 0
output_num = convert_images("google_images", output_num)
output_num = convert_images("taken_pictures", output_num)
output_num = convert_images("generation_images", output_num)