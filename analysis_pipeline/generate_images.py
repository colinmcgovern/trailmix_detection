import numpy
import cv2
import os

from random import choice
from random import seed
from random import random

from PIL import Image

sizes = [256,512,1024]

import xml.dom.minidom
import xml.etree.ElementTree as ET

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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

subimages = {}

for file in os.listdir('generation_subimages'):

	label = ''.join(i for i in file if not i.isdigit())
	label = label.replace(".png","")

	if label in subimages:
		subimages[label] = subimages[label] + 1
	else:
		subimages[label] = 1

seed(1)

image_iter = 0

for background_file in os.listdir('generation_backgrounds'):
	for width in sizes:
		for height in sizes:

			bounding_boxes = []

			background = cv2.imread("generation_backgrounds/"+background_file) 

			background = cv2.resize(background,(height,width))

			x_iter = 0
			y_iter = 0

			largest_img_height = 0

			while y_iter < height:

				label, count = choice(list(subimages.items()))
				img_index = int(random() * count + 1)

				img = cv2.imread("generation_subimages/"+label+"{}".format(img_index)+".png",cv2.IMREAD_UNCHANGED)

				img_width = int(50 * random() + 32)
				img_height = int(img_width * (float(img.shape[0]) / float(img.shape[1])))

				img = cv2.resize(img,(img_height,img_width))

				num_rotations = int(random() * 4)
				for i in range(num_rotations):
					img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

				img_width = img.shape[0]
				img_height = img.shape[1]

				if(largest_img_height < img_height):
					largest_img_height = img_height

				#placing subimage on image

				if(label!="herring" and x_iter+img_width<width and y_iter+img_height<height):
					bounding_boxes.append(BoundingBox(x1=int(y_iter), y1=int(x_iter),
					 x2=int(y_iter+img_height), y2=int(x_iter+img_width), label=label))

				# cv2.imwrite("generation_images/"+label+str(img_index)+"_image{}.png".format(image_iter),img)

				offset = numpy.array((x_iter,y_iter))

				if(x_iter+img_width < width and y_iter+img_height < height):
					for i in range(img_width):
						for j in range(img_height):

							if(img[i][j][3]):
								background[int(i+x_iter)][int(j+y_iter)] = \
								 img[int(i)][int(j)][:3]

				#iterating cursor
				x_iter = x_iter + img_width + random() * 128

				if(x_iter >= width):
					x_iter = 0
					y_iter = y_iter + largest_img_height

			cv2.imwrite("generation_images/image{}.jpg".format(image_iter),background)
			save_pascal_xml("generation_images",
				"image{}.xml".format(image_iter),background,bounding_boxes)

			image_iter = image_iter + 1

