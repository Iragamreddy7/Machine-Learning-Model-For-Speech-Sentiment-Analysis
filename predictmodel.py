# loading json and creating model
from keras.models import model_from_json

import os
import pandas as pd
import librosa
import glob
import keras
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def graph(alc):
	acc = ["Neutral", "Calm", "Happy", "Sad","Angry","Fearful", "Disgust","Surprised"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
 
	#Barplot for the dependent variable
	fig = plt.figure(0)
	plt.bar(acc,alc,color=colors)
	plt.xlabel('Emotion')
	plt.ylabel('Probabilty')
	plt.title("Emotion Chart")
	plt.draw() 
	plt.pause(0.01) #is necessary for the plot to update for some reason
	plt.clf()

def process(path):
	print("process")
	json_file = open('saved_models/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
	print("Loaded model from disk")
	opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6) 
	# evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
	sample_rate = np.array(sample_rate)
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
	featurelive = mfccs
	livedf2 = featurelive
	livedf2= pd.DataFrame(data=livedf2)
	livedf2 = livedf2.stack().to_frame().T
	twodim= np.expand_dims(livedf2, axis=2)
	livepreds = loaded_model.predict(twodim,batch_size=32,verbose=1)

	y_pred=livepreds.argmax(axis=1)
	print(y_pred)

	ac=[]
	for n in range(0,8):
		print(n)
		if n == y_pred:
			ac.append(10)
		else:
			ac.append(1)
	print(ac)
	graph(ac)
	#[1-neutral,2-calm,3-happy,4-sad,5-angry,6-fearful,7-disgust,8-surprised]
	labels = ["Neutral", "Calm", "Happy", "Sad","Angry","Fearful", "Disgust","Surprised"]
	print(y_pred, labels[int(y_pred)])
	return labels[int(y_pred)]
	

	