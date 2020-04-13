import warnings
import pyrebase
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from utils import base64_to_pil_image, pil_image_to_base64
import re
import cv2
import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api


app = Flask(__name__)
api = Api(app)
 
class measureSPO2(Resource):
	def get(self, user_index):

		config = {
			"apiKey": "AIzaSyDlmyLHYxZYCdmlI-pzKwkudQ85jdydBJ4",
		    "authDomain": "televital-hack.firebaseapp.com",
		    "databaseURL": "https://televital-hack.firebaseio.com",
		    "projectId": "televital-hack",
		    "storageBucket": "televital-hack.appspot.com",
		    "messagingSenderId": "209026679607",
		    "appId": "1:209026679607:web:68ae56edcb1abae7f290b2",
		    "measurementId": "G-07L3WDKM0H"

		}

		firebase = pyrebase.initialize_app(config)
		db = firebase.database()
		
		count = 0
		i=0
		A=100
		B=5
		bo = 0.0
		nm = db.child(user_index).get()
		res= nm.val()
		frame_text = res['spbase64']
		# convert it to a pil image
		input_img = base64_to_pil_image(frame_text)


		input_img = input_img.resize((320,240))

		img  = cv2.cvtColor(np.array(input_img), cv2.COLOR_BGR2RGB)

		#Red channel operations
		red_channel = img[:,:,2]
		mean_red = np.mean(red_channel)
		#print("RED MEAN", mean_red)
		std_red = np.std(red_channel)
		#print("RED STD", std_red)
		red_final = std_red/mean_red
		#print("RED FINAL",red_final)


		#Blue channel operations
		blue_channel = img[:,:,0]
		mean_blue = np.mean(blue_channel)
		#print("BLUE MEAN", mean_blue)
		std_blue = np.std(red_channel)
		#print("BLUE STD", std_blue)
		blue_final = std_blue/mean_blue
		#print("BLUE FINAL",blue_final)


		sp = A-(B*(red_final/blue_final))
		#print("SP_VALUE",sp)
		bo = bo + sp



		#this is the value to be returned on the result screen
		bo = bo/100.0	

		
		db.child(user_index).update({"sp":sp})
		return (1)

		# return (sp)	# yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		
		

api.add_resource(testabusive, '/spo/<user_index>')

if __name__ == '__main__':
   app.run()
