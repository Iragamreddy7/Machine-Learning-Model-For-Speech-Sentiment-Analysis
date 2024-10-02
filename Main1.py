import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import sqlite3
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from playsound import playsound

import trainmodel as trm
import predictmodel as pm
# Import necessary libraries 
from pydub import AudioSegment 

bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"

def Home():
	global window
	def clear():
	    print("Clear1")
	    txt1.delete(0, 'end')    



	window = tk.Tk()
	window.title("Emotion Recognition From Speech")

 
	window.geometry('1280x720')
	window.configure(background=bgcolor)
	#window.attributes('-fullscreen', True)

	window.grid_rowconfigure(0, weight=1)
	window.grid_columnconfigure(0, weight=1)
	

	message1 = tk.Label(window, text="Speech Sentiment Analysis" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
	message1.place(x=100, y=20)

	lbl = tk.Label(window, text="Select Input",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl.place(x=100, y=200)
	
	txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt.place(x=400, y=215)


	#lbl1 = tk.Label(window, text="Test Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	#lbl1.place(x=100, y=300)
	
	#txt1 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	#txt1.place(x=400, y=315)

	lbl4 = tk.Label(window, text="Notification : ",width=20  ,fg=fgcolor,bg=bgcolor  ,height=2 ,font=('times', 15, ' bold underline ')) 
	lbl4.place(x=100, y=400)

	message = tk.Label(window, text="" ,bg="white"  ,fg="black",width=30  ,height=3, activebackground = bgcolor ,font=('times', 15, ' bold ')) 
	message.place(x=400, y=400)

	def browse():
		res = ""
		message.configure(text= res)
		path=filedialog.askopenfilename()
		print(path)
		txt.delete(0, 'end')
		txt.insert('end',path)
		if path !="":
			print(path)
		else:
			res = "Select Input File"
			message.configure(text= res)
			tm.showinfo("Input error", "Select Input File")	


	def train():
		res = ""
		message.configure(text= res)
		trm.process()
		res = "Training Successfully Finished"
		message.configure(text= res)

		tm.showinfo("Input", "Training Successfully Finished")
	
	def predict():
		sym=txt.get()
		res = ""
		message.configure(text= res)

		if sym != "":
			# Input audio file to be sliced 
			audio = AudioSegment.from_wav(sym) 
			n = len(audio) 
			counter = 1

			interval = 3 * 1000
			overlap = 0
			# Initialize start and end seconds to 0 
			start = 0
			end = 0

			# Flag to keep track of end of file. 
			# When audio reaches its end, flag is set to 1 and we break 
			flag = 0

			# Iterate from 0 to end of the file, 
			# with increment = interval 
			for i in range(0, 2 * n, interval): 
				
				# During first iteration, 
				# start is 0, end is the interval 
				if i == 0: 
					start = 0
					end = interval 
			
				# All other iterations, 
				# start is the previous end - overlap 
				# end becomes end + interval 
				else: 
					start = end - overlap 
					end = start + interval 
		
				# When end becomes greater than the file length, 
				# end is set to the file length 
				# flag is set to 1 to indicate break. 
				if end >= n: 
					end = n 
					flag = 1
		
				# Storing audio file from the defined start to end 
				chunk = audio[start:end] 


				# Filename / Path to store the sliced audio 
				filename = 'output/chunk'+str(counter)+'.wav'

				# Store the sliced audio file to the defined path 
				chunk.export(filename, format ="wav") 
				# Print information about the current chunk 
				print("Processing chunk "+str(counter)+". Start = "+str(start)+" end = "+str(end))
				# Increment counter for the next chunk 
				counter = counter + 1
				try:
					mm(filename)
				except:
					pass
		else:		
			res = "Select Input File"
			message.configure(text= res)
			tm.showinfo("Input error", "Select Input File")

	def mm(filename):
		
		playsound(filename)
		res=pm.process(filename)
		#message.configure(text= res)
		#tm.showinfo("Input", "Predicted Successfully Finished")
		

	def playaudio():
		sym=txt.get()
		res = ""
		message.configure(text= res)

		if sym != "":
			playsound(sym)
			spf = wave.open(sym,'r')
			#Extract Raw Audio from Wav File
			signal = spf.readframes(-1)
			signal = np.fromstring(signal, 'Int16')
			plt.figure(1)
			plt.title('Signal Wave...')
			plt.plot(signal)
			plt.pause(5)
			plt.show(block=False)
			plt.close()			
		else:
			res = "Select Input File"
			message.configure(text= res)
			tm.showinfo("Input error", "Select Input File")


	browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	browse.place(x=650, y=200)
	
	play = tk.Button(window, text="Play Audio", command=playaudio  ,fg=fgcolor   ,bg=bgcolor1   ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	play.place(x=400, y=600)
	 
	process = tk.Button(window, text="Train Model", command=train  ,fg=fgcolor   ,bg=bgcolor1   ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	process.place(x=600, y=600)

	pro = tk.Button(window, text="Predict", command=predict  ,fg=fgcolor   ,bg=bgcolor1   ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	pro.place(x=800, y=600)

	quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	quitWindow.place(x=1000, y=600)

	window.mainloop()
Home()

