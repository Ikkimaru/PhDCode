import soundfile as sf
import librosa
import os

directory = "data_augment_same_actor/"

dir = sorted(os.listdir(directory))

for folder in dir:
	newFolder = folder.split('_')
	userNumber = int(newFolder[1])
	addDirectory = directory + folder
	wavFiles = sorted(os.listdir(addDirectory + "/"))

	for wavFile in wavFiles:
		fileDirectory = addDirectory + "/" + wavFile
		y, sr = librosa.load(fileDirectory)
		yt = librosa.effects.pitch_shift(y, sr, n_steps=4)		#Shift up by a major third (four half-steps) ///librosa.effects.pitch_shift
		newWav = wavFile.split("-")	
		finalWav = ""
		for number in newWav[:-2]: #03-01-01-01-01-01-01.wav
			finalWav += number + "-"


		finalWav += "0" + str(int(newWav[5]) + 2) + "-"
		if(userNumber < 10):
			finalWav += "0" + str(userNumber) + ".wav"
		else:
			finalWav += str(userNumber) + ".wav"
		sf.write(addDirectory + "/" + finalWav, yt, 16000)

	print("Actor " + str(userNumber) + "/24" + " Completed")




	#RENAME FOLDERS
	#newFolder = folder.split('_')
	#newFolder[1] = int(newFolder[1]) + 24
	#print(newFolder[0] + "_" + str(newFolder[1]))
#y, sr = librosa.load("OriginalTest.wav")
#yt = librosa.effects.pitch_shift(y, sr, n_steps=4)
#sf.write('OriginalTest_Shift.wav', yt, 16000)