import soundfile as sf
import librosa
import os

directory = "data_augment/"

dir = sorted(os.listdir(directory))

for folder in dir:
	newFolder = folder.split('_')
	userNumber = int(newFolder[1]) + 24
	addDirectory = directory + "Actor_" + str(userNumber)
	os.mkdir(addDirectory)

	for wavFile in sorted(os.listdir(directory+folder+"/")):
		fileDirectory = directory+folder+"/" + wavFile
		y, sr = librosa.load(fileDirectory)
		yt = librosa.effects.pitch_shift(y, sr, n_steps=4)		#Shift up by a major third (four half-steps) ///librosa.effects.pitch_shift
		newWav = wavFile.split("-")	
		finalWav = ""
		for number in newWav[:-1]: #03-01-01-01-01-01-01.wav
			finalWav += number + "-"

		finalWav += str(userNumber) + ".wav"
		sf.write(addDirectory + "/" + finalWav, yt, 16000)

	print("Actor " + str(userNumber) + "/48" + " Completed")




	#RENAME FOLDERS
	#newFolder = folder.split('_')
	#newFolder[1] = int(newFolder[1]) + 24
	#print(newFolder[0] + "_" + str(newFolder[1]))
#y, sr = librosa.load("OriginalTest.wav")
#yt = librosa.effects.pitch_shift(y, sr, n_steps=4)
#sf.write('OriginalTest_Shift.wav', yt, 16000)