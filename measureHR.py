import warnings
import pyrebase
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from utils import base64_to_pil_image, pil_image_to_base64
import re
import cv2
import numpy as np
from numpy import mean
from flask import Flask, request
from flask_restful import Resource, Api


app = Flask(__name__)
api = Api(app)
 
class measureHR(Resource):
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
		nm = db.child(user_index).get()
		res= nm.val()
		video_frames = res['hrbase64']

		video_strings = video_frames.split(';')
		video_strings = video_strings[3:]

		def buildGauss(frame, levels):
			pyramid = [frame]
			for level in range(levels):
				frame = cv2.pyrDown(frame)
				pyramid.append(frame)
			return pyramid
		def reconstructFrame(pyramid, index, levels):
			fiFrame = pyramid[index]
			for level in range(levels):
				fiFrame = cv2.pyrUp(fiFrame)
			fiFrame = fiFrame[:videoHeight, :videoWidth]
			return fiFrame

		def applyFFT(frames, fps):
			n = frames.shape[0]
			t = np.linspace(0,float(n)/fps, n)
			disp = frames.mean(axis = 0)
			y = frames - disp

			k = np.arange(n)
			T = n/fps
			frq = k/T # two sides frequency range
			freqs = frq[range(n//2)] # one side frequency range

			Y = np.fft.fft(y, axis=0)/n # fft computing and normalization
			signals = Y[range(n//2), :,:]
			
			return freqs, signals

		def bandPass(freqs, signals, freqRange):

			signals[freqs < freqRange[0]] *= 0
			signals[freqs > freqRange[1]] *= 0

			return signals

		def find(condition):
			res, = np.nonzero(np.ravel(condition))
			return res


		def freq_from_crossings(sig, fs):
			"""Estimate frequency by counting zero crossings
    
			"""
			#print(sig)
			# Find all indices right before a rising-edge zero crossing
			indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
			x = sig[1:]
			x = mean(x)

			
			return x

		def searchFreq(freqs, signals, frames, fs):

			curMaximumval = 0
			freMax = 0
			Mi = 0
			Mj = 0
			for i in range(10, signals.shape[1]):
				for j in range(signals.shape[2]):

					idxMaximumval = abs(signals[:,i,j])
					idxMaximumval = np.argmax(idxMaximumval)
					freqMaximumval = freqs[idxMaximumval]
					ampMaximumval = signals[idxMaximumval,i,j]
					c, a = abs(curMaximumval), abs(ampMaximumval)
					if (c < a).any():
						curMaximumval = ampMaximumval
						freMax = freqMaximumval
						Mi = i
						Mj = j
                # print "(%d,%d) -> Freq:%f Amp:%f"%(i,j,freqMaximumval*60, abs(ampMaximumval))

			y = frames[:,Mi, Mj]
			y = y - y.mean()
			fq = freq_from_crossings(y, fs)
			rate_fft = freMax*60
			
			rate_count = round(20+(fq*10))

			if np.isnan(rate_count):
				rate = rate_fft
			elif abs(rate_fft - rate_count) > 20:
				rate = rate_fft
			else:
				rate = rate_count

			return rate

		def answer(videoStrings):

			sampleLength = 10
			firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
			firstGauss = buildGauss(firstFrame, levels+1)[levels]
			sample = np.zeros((sampleLength, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
		
			idx = 0
			
			respRate = []	

			#pipeline = PipeLine(videoFrameRate)
			for i in range(len(videoStrings)):
				input_img = base64_to_pil_image(videoStrings[i])

				input_img = input_img.resize((320,240)) 

				frame  = cv2.cvtColor(np.array(input_img), cv2.COLOR_BGR2RGB)
		
				detectionFrame = frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):int(realWidth-int(videoWidth/2)), :]


				sample[idx] = buildGauss(detectionFrame, levels+1)[levels]
			
				freqs, signals = applyFFT(sample, videoFrameRate)
				signals = bandPass(freqs, signals, (0.2, 0.8))
				respiratoryRate = searchFreq(freqs, signals, sample, videoFrameRate)

				#frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):(realWidth-int(videoWidth/2)), :] = outFrame
				
				idx = (idx + 1) % 10 		

				respRate.append(respiratoryRate)

			l = []
			a = max(respRate)
			b = mean(respRate)
			if b < 0:
				b = 5
			l.append(a)
			l.append(b)


			return mean(l)	


		# Webcam Parameters
		realWidth = 320
		realHeight = 240
		videoWidth = 160
		videoHeight = 120
		videoChannels = 3
		videoFrameRate = 15


		# Color Magnification Parameters
		levels = 3
		alpha = 170
		minFrequency = 1.0
		maxFrequency = 2.0
		bufferSize = 150
		bufferIndex = 0

		# Output Display Parameters
		font = cv2.FONT_HERSHEY_SIMPLEX
		loadingTextLocation = (20, 30)
		bpmTextLocation = (videoWidth//2 + 5, 30)
		fontScale = 1
		fontColor = (0,0,0)
		lineType = 2
		boxColor = (0, 255, 0)
		boxWeight = 3

		# Initialize Gaussian Pyramid
		firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
		firstGauss = buildGauss(firstFrame, levels+1)[levels]
		videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
		fourierTransformAvg = np.zeros((bufferSize))

		# Bandpass Filter for Specified Frequencies
		frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
		mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

		# Heart Rate Calculation Variables
		bpmCalculationFrequency = 15
		bpmBufferIndex = 0
		bpmBufferSize = 10
		bpmBuffer = np.zeros((bpmBufferSize))
		i = 0
		bpm_values = []
		for j in range(len(video_strings)):
			# convert it to a pil image
			input_img = base64_to_pil_image(video_strings[j])

			input_img = input_img.resize((320,240))

			img  = cv2.cvtColor(np.array(input_img), cv2.COLOR_BGR2RGB)

			detectionFrame = img[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):int(realWidth-int(videoWidth/2)), :]

			# Construct Gaussian Pyramid
			videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
			fourierTransform = np.fft.fft(videoGauss, axis=0)
			# Bandpass Filter
			fourierTransform[mask == False] = 0

			# Grab a Pulse
			if bufferIndex % bpmCalculationFrequency == 0:
				i = i + 1
				for buf in range(bufferSize):
					fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
				hz = frequencies[np.argmax(fourierTransformAvg)]
				bpm = 60.0 * hz
				bpmBuffer[bpmBufferIndex] = bpm
				# print("BPM Buffer List: ", bpmBuffer)
				bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

			# Amplify
			filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
			filtered = filtered * alpha

			# Reconstruct Resulting Frame
			fiFrame = reconstructFrame(filtered, bufferIndex, levels)
			outFrame = detectionFrame + fiFrame
			outFrame = cv2.convertScaleAbs(outFrame)

			bufferIndex = (bufferIndex + 1) % bufferSize
			
			if i > bpmBufferSize:
				bpm_values.append(bpmBuffer.mean())



		# take the maximum val of the calculated heart rates
		hr = max(bpm_values)
		print(hr)

		# call the function to calculate respiratory rate
		rr = answer(video_strings)
		print(rr)
		

		# push the data to database
		db.child(user_index).update({"hr":hr,'rr':rr})
		return (1)


api.add_resource(testabusive, '/hr/<user_index>')

if __name__ == '__main__':
   app.run()