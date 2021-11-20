from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import urllib.request
import sys, os, shutil
from werkzeug.utils import secure_filename
import torch
from yolov5.custom_detect import run
import cv2
import random
from time import sleep
import json

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	 
def draw_boxes(img,pred,org_h,org_w):
	(h, w) = img.shape[:2]

	for obj in pred[0]:

		corner1 = (int(obj[0]/org_w * w),
		 int(obj[1]/org_h * h))
		corner2 = (int(obj[2]/org_w * w),
		 int(obj[3]/org_h * h))

		color = (0,0,0)
		letter = '?'

		if(obj[5] == 0.):
			color = (255,0,255)
			letter = 'A'
		elif(obj[5] == 1.):
			color = (0,255,0)
			letter = 'R'
		elif(obj[5] == 2.):
			color = (0,0,255)
			letter = 'M'
		else:
			color = (255,255,0)
			letter = "P"

		letter = letter + " " + str(int(obj[4]*100)) + "%"

		img = cv2.rectangle(img, corner1, corner2, color, -1)

		img = cv2.putText(img, letter, (int((corner1[0]+corner2[0])/2)-8,
			int((corner1[1]+corner2[1])/2))
		  , cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)

	return img

def pred_to_dict(pred):

	count_dict = {
		"mm": 0,
		"peanut": 0,
		"raisin": 0,
		"almond": 0
	}

	for obj in pred[0]:
		if(obj[5] == 0.):
			count_dict["almond"] = count_dict["almond"] + 1
		elif(obj[5] == 1.):
			count_dict["raisin"] = count_dict["raisin"] + 1
		elif(obj[5] == 2.):
			count_dict["mm"] = count_dict["mm"] + 1
		else:
			count_dict["peanut"] = count_dict["peanut"] + 1

	return count_dict

def write_description(pred):

	count_dict = pred_to_dict(pred)

	description = "There"

	if(count_dict["mm"] == 1):
		description += " is 1 m and m"
	else:
		description += " are {} m and ms".format(count_dict["mm"])

	if(count_dict["peanut"] == 1):
		description += ", 1 peanut"
	else:
		description += ", {} peanuts".format(count_dict["peanut"])

	if(count_dict["raisin"] == 1):
		description += ", 1 raisin"
	else:
		description += ", {} raisins".format(count_dict["raisin"])

	if(count_dict["almond"] == 1):
		description += ", and 1 almond."
	else:
		description += ", and {} almonds.".format(count_dict["almond"])

	return description

def clear_folder(folder):
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete {}'.format(e))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/result/')
def result():
	return render_template('result.html')
 
@app.route('/', methods=['POST'])
def upload_image():

	clear_folder('runs')
	clear_folder('static/uploads')

	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)

	uploaded_files = request.files.getlist("file")

	unlabeled_filenames = []
	labeled_filenames = []
	descriptions = []
	counts = []

	for file in uploaded_files:

		if file.filename == '':
			flash('No image selected for uploading')
			return redirect(request.url)

		if file and allowed_file(file.filename):
			
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

			#Run algorithm
			img, pred = run(UPLOAD_FOLDER+filename)

			#Setting up display image
			org_image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename), cv2.IMREAD_COLOR)
			(h, w) = org_image.shape[:2]
			display_image_width = 512
			ratio = h / w
			display_image_height = int(display_image_width * ratio)
			dim = (display_image_width, display_image_height)
			display_image = cv2.resize(org_image, dim)

			#Assigning ID
			index = random.randint(0,1E8)

			#Saving images
			cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "unlabeled_{}.jpg".format(index)),display_image)
			cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "labeled_{}.jpg".format(index)), draw_boxes(display_image,pred,h,w))

			#Adding to output array			
			unlabeled_filenames.append("unlabeled_{}.jpg".format(index))
			labeled_filenames.append("labeled_{}.jpg".format(index))
			descriptions.append(write_description(pred))
			counts.append(json.dumps(pred_to_dict(pred)))

		else:
			flash('Allowed image types are - png, jpg, jpeg, gif')
			return redirect(request.url)

	return render_template('result.html',
	data=zip(unlabeled_filenames,labeled_filenames,descriptions,counts))
 
@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
	app.run(port=80)