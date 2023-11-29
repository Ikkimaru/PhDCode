#Website
#https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn

print("Importing")

import xlwt
from xlwt import Workbook
import os.path
from os import path
from numpy import random
import numpy
import time # Timer
from datetime import timedelta
from datetime import datetime
import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.model_selection import cross_val_score # Cross validation
from sklearn import preprocessing # For KNN

from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
#NEW MODELS
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn import svm #Support Vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.naive_bayes import GaussianNB #Naive Bayes
#from hmmlearn import hmm #Hidden Markov Model

from sklearn.metrics import accuracy_score # to measure how good we are
from sklearn.metrics import classification_report, confusion_matrix # Self Explanitory
from sklearn.utils.multiclass import unique_labels

import warnings
warnings.filterwarnings("ignore")


#Database is default, no automation at this point

#Iterations: Models (MLP, Random Forest), Features(MFCC, Chroma) {12345, 2345, 3451, 4512, 5123}
#Variables: Amount of tests per experiment, Accuracy Split, Emotions(Happy, Sad)
#Need to capture (Headings): Experiment num, features, amount of features, T* accuracy, T* time, Average accuracy, lowest accuracy, highest accuracy.
#TODO: Model Parameters (For example MLP:alpha, batch size)

# VARIABLES HERE
actorFolderLocation = "C:/Users/Ebenv/Desktop/ConvertFromJupyter/data/Actor_*/*.wav"
wb = Workbook()
fileName = 'example.xls'
originalFileName = path.splitext(fileName)[0]
duplicateValue = 1
while path.exists(fileName):
	fileName = originalFileName + "(" + str(duplicateValue) + ").xls"
	duplicateValue += 1

#wb.save(fileName)
#models = ["MLP", "Random Forest", 'Naive Bayes'] # This will dictate the amount of experiments
models = "Random Forest" # This will dictate the amount of experiments
#features = ["MFCC", "Chroma", "Mel", "Contrast", "Tonnetz"] # This will dictate the amount of experiments
features = ["MFCC", "Chroma", "Mel", "Contrast", "Tonnetz"] # This will dictate the amount of experiments
AVAILABLE_EMOTIONS = ["calm", "happy", "sad", "disgust", "angry", "fearful", "surprised"]
#selectedEmotions = ["neutral", "calm", "happy", "sad", "disgust", "angry", "fearful", "surprised"]
selectedEmotions = ["disgust", "fearful", "surprised"]	# This is for utility calculation
accuracySplit = [75, 25] # Train/Test  (Maybe iterate Later)
amountOfTests = 5 # Amount of tests for each experiment
kFoldAccuracy = True
excelCursor = [0, 0]
knnX, knny = [], []


# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('Sheet 1') #THIS IS HARD CODED, ONLY ONE SHEET PER RUN ('Sheet 1', cell_overwrite_ok=True)
#sheet1.write(1, 0, 'Name') Row, Column


def print_Excel(value):
	sheet1.write(excelCursor[0], excelCursor[1], value)

def move_Cursor(direction, amount):
	if direction == "row":
		excelCursor[0] = excelCursor[0] + amount
	if direction == "column":
		excelCursor[1] = excelCursor[1] + amount

def new_Row_Left():
	move_Cursor('row', 1)
	excelCursor[1] = 0

def print_Cursor_Position():
	print("Cursor Position: Row " + str(excelCursor[0]) + " / Column " + str(excelCursor[1]))


def extract_feature(file_name, featureList):
	"""
	Extract feature from audio file `file_name`
		Features supported:
			- MFCC (mfcc)
			- Chroma (chroma)
			- MEL Spectrogram Frequency (mel)
			- Contrast (contrast)
			- Tonnetz (tonnetz)
		e.g:
		`features = extract_feature(path, mel=True, mfcc=True)`
	"""
	#mfcc = kwargs.get("mfcc")
	#chroma = kwargs.get("chroma")
	#mel = kwargs.get("mel")
	#contrast = kwargs.get("contrast")
	#tonnetz = kwargs.get("tonnetz")

	mfcc = "MFCC" in featureList
	chroma = "Chroma" in featureList
	mel = "Mel" in featureList
	contrast = "Contrast" in featureList
	tonnetz = "Tonnetz" in featureList


	with soundfile.SoundFile(file_name) as sound_file:
		X = sound_file.read(dtype="float32")
		sample_rate = sound_file.samplerate
		if chroma or contrast:
			stft = np.abs(librosa.stft(X))
		result = np.array([])
		if mfcc:
			mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
			result = np.hstack((result, mfccs))
		if chroma:
			chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
			result = np.hstack((result, chroma))
		if mel:
			#mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
			mel = np.mean(librosa.feature.melspectrogram(y=X))
			result = np.hstack((result, mel))
		if contrast:
			contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
			result = np.hstack((result, contrast))
		if tonnetz:
			tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
			result = np.hstack((result, tonnetz))
	return result

def aggregated_confusion_matrix(y_test, y_pred, selectedEmotion):

	TruePositive = 0 		#Warn/Warn are warn emotions correctly identified as warn (Fearful identified as Disgust is still correct)
	FalseNegative = 0 		#Warn/Ignore are warn emotions incorrectly identified as ignore (Fearful identified as happy is wrong)
	FalsePositive = 0 		#Ignore/Warn are ignore emotions incorrectly identified as warn (Happy identified as fearful is wrong)
	TrueNegative = 0 		#Ignore/Ignore are ignore emotions correctly identified as ignore (Happy identified as calm is still correct)
	#PredictedPositive = 0 	#TruePositive(W/W) + FalsePositive(I/W)
	#PredictedNegative = 0 	#FalsePositive(W/I) + TrueNegative(I/I)
	#UtilityValue = 0 		#(PredictedPositive[TP(W/W)+FP(I/W)] + TrueNegative(I/I))/(PredictedPositive[TP(W/W)+FP(I/W)] + PredictedNegative[FN(W/I) + TN(I/I)])
	ActualNegative = 0 		#Total number of samples for non-selected emotions
	ActualPositive = 0 		# Total number of samples for selected emotions

	
	emotionLabels = unique_labels(y_test, y_pred)
	ignoredEmotions = [0] * (len(emotionLabels) + 1)	# Create empty list with length of all emotions
	ignoredEmotions[0] = "IgnoredEmotions"
	confusionMatrixResults = confusion_matrix(y_test, y_pred)
	warnedEmotions = []
	warnedEmotionsIndex = []

	for index, emotion in enumerate(emotionLabels, start=1):	# Go through all emotions
		if emotion in selectedEmotions:
			warnedEmotionsIndex.append(index)	# Add index of waned emotions

	for index, row in enumerate(confusionMatrixResults):
		buildEmotion = []
		if emotionLabels[index] in selectedEmotions:	# If these are part of the selected emotions
			buildEmotion.append(str(emotionLabels[index]))	# Add emotion label
			for warnIndex, item in enumerate(row, start=1):
				buildEmotion.append(str(item))	# Add row for the selected emotion
				ActualPositive += int(item)
				if warnIndex in warnedEmotionsIndex:
					TruePositive += int(item)	# Warned emotions correctly warned
				else:
					FalseNegative += int(item)	# Warned emotions incorrectly ignored
				
			warnedEmotions.append(buildEmotion)	# Add row into bigger collection
		else:
			for ignoreIndex, item in enumerate(row, start=1):
				ignoredEmotions[ignoreIndex] = int(ignoredEmotions[ignoreIndex]) + int(item) # Add the rest of the unselected emotions into ignore list
				ActualNegative += int(item)
				if ignoreIndex in warnedEmotionsIndex:
					FalsePositive += int(item)	# Ignore emotions incorrectly warned
				else:
					TrueNegative += int(item)	# Ignore emotions correctly Ignored
	

	#print("Actual Positive: " + str(ActualPositive))
	#print("Actual Negative: " + str(ActualNegative))
	#print("True Positive: " + str(TruePositive))
	#print("False Positive: " + str(FalsePositive))
	#print("True Negative: " + str(TrueNegative))
	#print("False Negative: " + str(FalseNegative))
	return ActualPositive, ActualNegative, TruePositive, FalsePositive, TrueNegative, FalseNegative;

def calculate_utility_value(TruePositive, FalsePositive, TrueNegative, FalseNegative):
	# (PP + TN)/(PP + PN)
	predictedPositive = TruePositive + FalsePositive
	predictedNegative = FalseNegative + TrueNegative

	return ((predictedPositive+TrueNegative)/(predictedPositive+predictedNegative))*100


# all emotions on RAVDESS dataset

int2emotion = {
	"01": "neutral",
	"02": "calm",
	"03": "happy",
	"04": "sad",
	"05": "angry",
	"06": "fearful",
	"07": "disgust",
	"08": "surprised"
}

"""
# ALL EMOTIONS
AVAILABLE_EMOTIONS = {
	"neutral",
	"calm",
	"happy",
	"sad",
	"angry",
	"fearful",
	"disgust",
	"surprised"
}

# we allow only these emotions ( feel free to tune this on your need )
# BIG TABLE TESTS
AVAILABLE_EMOTIONS = {
	"angry",
	"sad",
	"neutral",
	"happy"
}

DEFAULT-->
AVAILABLE_EMOTIONS = {
	"neutral",
	"happy",
	"sad",
	"calm",
	"fearful"
}

SONG ONLY EMOTIONS
AVAILABLE_EMOTIONS = {
	"calm",
	"happy",
	"sad",
	"angry",
	"fearful"
}

CHAPTER 7 BASELINE TESTS
AVAILABLE_EMOTIONS = {
	"angry",
	"sad",
	"surprised",
	"fearful"
}
AVAILABLE_EMOTIONS = {
	"calm",
	"happy",
	"sad",
	"angry",
	"fearful"
}
"""

def load_data(test_size, featureList):
	X, y = [], []
	knnX.clear()
	knny.clear()
	for file in glob.glob(actorFolderLocation):
		# get the base name of the audio file
		basename = os.path.basename(file)
		songSpeech = basename.split("-")[1] # This Identifies Song of Speech
		onlyNumber = basename.split("-")[6] # This is gender
		onlyNumber = onlyNumber.split(".")[0] # This is to remove .mp3 at end of file
		#if (int(onlyNumber)%2 != 0): #Only testing Male
		#if (int(onlyNumber)%2 == 0): #Only testing Female
		# get the emotion label
		emotion = int2emotion[basename.split("-")[2]]
		# we allow only AVAILABLE_EMOTIONS we set
		if emotion not in AVAILABLE_EMOTIONS:
			continue
		# extract speech features
		features = extract_feature(file, featureList)
		# add to data
		X.append(features)
		knnX.append(features)


		#if(songSpeech == "01"):
			#y.append(emotion + ";Speech")
		#else:
			#y.append(emotion + ";Song")

		y.append(emotion)
		knny.append(emotion)
	# split the data to training and testing and return it
	return train_test_split(np.array(X), y, test_size=test_size, random_state=7)




experimentNumber = 1

##########      Headings
print_Excel(models + ": " + str(AVAILABLE_EMOTIONS))
move_Cursor('column', 1)

print_Excel("Accuracy Split: " + ','.join([str(elem) for elem in accuracySplit]))
move_Cursor('column', 1)

print_Excel("Selected Emotions: " + ','.join([str(elem) for elem in selectedEmotions]))
move_Cursor('column', 1)
new_Row_Left()

print_Excel("Name:")
move_Cursor('column', 1)

for featurename in features:
	print_Excel(featurename)
	move_Cursor('column', 1)

print_Excel("# Features")
move_Cursor('column', 1)

for i in range(1, amountOfTests + 1): # Starting from 1 and excluding final number
	if kFoldAccuracy:
		print_Excel("T" + str(i) + " KFold")
	else:
		print_Excel("T" + str(i) + " Accuracy")
	#print_Excel("T" + str(i) + " Train Time (s)")
	move_Cursor('column', 1)
	print_Excel("T" + str(i) + " Accuracy")
	move_Cursor('column', 1)
	print_Excel("Utility Value")
	move_Cursor('column', 1)
	print_Excel("T" + str(i) + " Matrix")
	move_Cursor('column', 1)


#print_Excel("Average Accuracy")
#move_Cursor('column', 1)

print_Excel("Utility Value")
move_Cursor('column', 1)

print_Excel("False Positive")
move_Cursor('column', 1)

print_Excel("False Negative")
move_Cursor('column', 1)

print_Excel("WarnCube")
move_Cursor('column', 1)

print_Excel("User Experience (How many warn is legit) WW/(WW+IW)")
move_Cursor('column', 1)
wb.save(fileName)
new_Row_Left()

t0 = time.time()
#while experimentNumber <= len(features):
while experimentNumber <= 1:		#Only 1 for testing								# Iterate over experiments

	
	
	averageHighest = 0
	averageLowest = 100
	print_Excel("Experiment " + str(experimentNumber))
	move_Cursor('column', 1)

	###########		Testing
	featureArrayNum = experimentNumber - 1 # Start feature at number of experiment (Experiment 2 would start at Chroma)
	featureCollection = []

	averageAccuracy = 0
	while featureArrayNum < len(features):					# Iterate over combinations of features (At the end)

		#These are for utility value calculation
		ActualPositive = 0
		ActualNegative = 0
		TruePositive = 0
		FalsePositive = 0
		TrueNegative = 0
		FalseNegative = 0

		print("Experiment: ",experimentNumber, " out of ", len(features), "\nFeature: ",featureArrayNum + 1, " out of ", len(features))
		featureCollection += [features[featureArrayNum]]

		i = 0
		while i < len(features):
			if features[i] in featureCollection:
				print_Excel("X")
			move_Cursor('column', 1)
			i += 1

		print("Loading Data")

		duration = str(timedelta(seconds=time.time() - t0))
		x = duration.split(':')
		print('Time in hh:mm:ss:', x[0], 'Hours', x[1], 'Minutes', x[2], 'Seconds')

		now = datetime.now()

		current_time = now.strftime("%H:%M:%S")
		print("Current Time =", current_time)

		# load RAVDESS dataset, 75% training 25% testing
		X_train, X_test, y_train, y_test = load_data(accuracySplit[1]/100, featureCollection)

		# WE TEST THE SPLIT HERE
		#print("Test: " + str(y_test))
		#print("Train Feature #: " + str(len(X_train)))
		#newYList = []#@@@@@@@@
		#newXList = []#@@@@@@@@
		#index = 0#@@@@@@@@
		#indexLength = len(y_test)#@@@@@@@@
		#historyList = []#@@@@@@@@
		#while index < indexLength:#@@@@@@@@
			#getSong = y_test[index].split(";")[1]#@@@@@@@@
			#if getSong == "Speech":#@@@@@@@@
				#newYList.append(y_test[index].split(";")[0])#@@@@@@@@
				#newXList.append(X_test[index])#@@@@@@@@
			#index += 1#@@@@@@@@

		# Cleaning the Train Emotions
		#index = 0#@@@@@@@@
		#indexLength = len(y_train)#@@@@@@@@
		#newYTrain = []#@@@@@@@@
		#while index < indexLength:#@@@@@@@@
			#newYTrain.append(y_train[index].split(";")[0])#@@@@@@@@
			#index += 1#@@@@@@@@


		# rewriting the old lists
		#y_test = newYList#@@@@@@@@
		#X_test = newXList#@@@@@@@@
		#y_train = newYTrain#@@@@@@@@


		#print("Test: " + str(newYList))
		#print("Train Feature #: " + str(len(newXList)))
		#print("\n\n")
		#print(y_train)
		#print(newYTrain)



		#Filter Train OR Test data based on emotion
		#trainSelectedEmotion = ["disgust","fearful","surprised"]
		#tempEmotionList = []
		#tempDataList = []
		#for filterIndex, emotion in enumerate(y_train):
			#if emotion in trainSelectedEmotion:
				#tempEmotionList.append(y_train[filterIndex])
				#tempDataList.append(X_train[filterIndex])

		#y_test = np.array(tempEmotionList)
		#X_test = np.array(tempDataList)


		print_Excel(X_train.shape[1])
		move_Cursor('column', 1)


		for i in range(1, amountOfTests +1): # Starting from 1 and excluding final number
			print("===================================================")
			print("Test :",i," out of ",amountOfTests)
			# best model, determined by a grid search
			model_params = {
				'alpha': 0.01,
				'batch_size': 256,
				'epsilon': 1e-08, 
				'hidden_layer_sizes': (300,), 
				'learning_rate': 'adaptive', 
				'max_iter': 1000, 
			}

			# initialize Multi Layer Perceptron classifier
			# with best parameters ( so far )
			if models == "MLP":
				model = MLPClassifier(**model_params) #MLP / Neural Net
			if models == "KNN":
				model = KNeighborsClassifier(n_neighbors=1, weights='uniform') #KNN
			if models == "SVML":
				model = svm.SVC(kernel='linear',max_iter=-1) # Support Vector Machine Linear Kernel
			if models == "SVMR":
				model = svm.SVC(kernel='rbf',max_iter=-1) # Support Vector Machine Radial Basis Function Kernel
			if models == "SVMP":
				model = svm.SVC(kernel='poly',max_iter=-1) # Support Vector Machine Poly Kernel
			if models == "Random Forest":
				model = RandomForestClassifier() #Random Forest
			if models == "Naive Bayes":
				model = GaussianNB() #Naive Bayes

			# train the model
			#t0 = time.time()
			print("[*] Training the model...")
			#X_train, X_test, y_train, y_test = train_test_split(np.array(knnX), knny, test_size=accuracySplit[1]/100) # THIS WAS FOR KNN
			model.fit(X_train, y_train)

			scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
			print(scores)
			print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

			# predict 25% of data to measure how good we are
			y_pred = model.predict(X_test)

			# calculate the accuracy
			accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
			#t1 = time.time()
			print("Accuracy: {:.2f}%".format(accuracy*100))
			#total = t1-t0
			#print("Duration: " + str(round(total)))
			print(classification_report(y_test, y_pred))
			print(confusion_matrix(y_test, y_pred))


			tempActualPositive, tempActualNegative, tempTruePositive, tempFalsePositive, tempTrueNegative, tempFalseNegative = aggregated_confusion_matrix(y_test, y_pred, selectedEmotions)
			ActualPositive += tempActualPositive
			ActualNegative += tempActualNegative
			TruePositive += tempTruePositive
			FalsePositive += tempFalsePositive
			TrueNegative += tempTrueNegative
			FalseNegative += tempFalseNegative

			

			print("Test :",i," out of ",amountOfTests)


			testResult = accuracy*100
			#if kFoldAccuracy:
			averageAccuracy += scores.mean()*100  # Kfold average
			print_Excel(scores.mean()*100) #KFold Accuracy
			move_Cursor('column', 1)
			#else:EBEN
			averageAccuracy += testResult
			print_Excel(testResult)

			#move_Cursor('column', 1)
			#averageAccuracy += testResult
			########This is for bit 10 tests
			#print_Excel(testResult)
			move_Cursor('column', 1)
			#########

			utilityValue = calculate_utility_value(tempTruePositive, tempFalsePositive, tempTrueNegative, tempFalseNegative)
			print_Excel(utilityValue)
			move_Cursor('column', 1)


			out_arr = numpy.array_str(confusion_matrix(y_test, y_pred))
			print_Excel(out_arr)
			#print_Excel(str(confusion_matrix(y_test, y_pred))
			move_Cursor('column', 1)
			#print_Excel(str(round(total)))
			#move_Cursor('column', 1)

			print("===================================================")



		averageAccuracy = averageAccuracy/amountOfTests
		print("AVERAGE: " + str(averageAccuracy) + "\n\n")
		#print_Excel(averageAccuracy)
		#move_Cursor('column', 1)


		# Find highest and lowest
		if averageAccuracy < averageLowest:
			averageLowest = averageAccuracy
		if averageAccuracy > averageHighest:
			averageHighest = averageAccuracy
		averageAccuracy = 0


		ActualPositive = ActualPositive/amountOfTests
		ActualNegative = ActualNegative/amountOfTests
		TruePositive = TruePositive/amountOfTests
		FalsePositive = FalsePositive/amountOfTests
		TrueNegative = TrueNegative/amountOfTests
		FalseNegative = FalseNegative/amountOfTests

		print(str(ActualPositive) + " / " + str(ActualNegative) + " / " + str(TruePositive) + " / " + str(FalsePositive) + " / " + str(TrueNegative) + " / " + str(FalseNegative))
		print("Warn/Warn: " + str(TruePositive))
		print("Warn/Ignore: " + str(FalseNegative))
		print("Ignore/Warn: " + str(FalsePositive))
		print("Ignore/Ignore: " + str(TrueNegative))
		print("\nPositive Support: " + str(ActualPositive) + " / Negative Support: " + str(ActualNegative))
		
		utilityValue = calculate_utility_value(TruePositive, FalsePositive, TrueNegative, FalseNegative)

		print("\nUtility Value: " + str(utilityValue) + "\n")
		print("===================================================")


		print_Excel(str(utilityValue))
		move_Cursor('column', 1)

		print_Excel(str((FalsePositive/ActualNegative)*100))
		#print_Excel("N/A")
		move_Cursor('column', 1)

		print_Excel(str((FalseNegative/ActualPositive)*100))
		#print_Excel("N/A")
		move_Cursor('column', 1)

		print_Excel("WW " + str(TruePositive) + ",WI " + str(FalseNegative) + ",IW " + str(FalsePositive) + ",II " + str(TrueNegative))
		move_Cursor('column', 1)

		print_Excel(str((TruePositive/(TruePositive+FalsePositive)*100)) + " %")
		#print_Excel("N/A")
		move_Cursor('column', 1)

		duration = str(timedelta(seconds=time.time() - t0))
		x = duration.split(':')
		sec = x[2].split('.')
		x[2] = sec[0]
		move_Cursor('column', 1)
		totalTime = 'Current Duration: ' + str(x[0]) + ' H ' + str(x[1]) + ' M ' + str(x[2]) + ' S '
		print_Excel(totalTime)

		new_Row_Left()
		move_Cursor('column', 1)
		wb.save(fileName)

		featureArrayNum += 1		# Signifies the end of that combo testing

	#R-R-R-R-Rewind!!!!

	featureArrayNum = 0	#Start loop over at zero
	while featureArrayNum < experimentNumber - 2:			# Iterate over combinations of features (Looped back if needed)


		#These are for utility value calculation
		ActualPositive = 0
		ActualNegative = 0
		TruePositive = 0
		FalsePositive = 0
		TrueNegative = 0
		FalseNegative = 0

		print("Experiment: ",experimentNumber, " out of ", len(features), "\nFeature: ",featureArrayNum + 1, " out of ", len(features))
		featureCollection += [features[featureArrayNum]]

		i = 0
		while i < len(features):
			if features[i] in featureCollection:
				print_Excel("X")
			move_Cursor('column', 1)
			i += 1


		print("Loading Data")


		duration = str(timedelta(seconds=time.time() - t0))
		x = duration.split(':')
		print('Time in hh:mm:ss:', x[0], 'Hours', x[1], 'Minutes', x[2], 'Seconds')

		now = datetime.now()

		current_time = now.strftime("%H:%M:%S")
		print("Current Time =", current_time)

		# load RAVDESS dataset, 75% training 25% testing
		X_train, X_test, y_train, y_test = load_data(accuracySplit[1]/100, featureCollection)

		# WE TEST THE SPLIT HERE
		#print("Test: " + str(y_test))
		#print("Train Feature #: " + str(len(X_train)))
		#newYList = []#@@@@@@@@
		#newXList = []#@@@@@@@@
		#index = 0#@@@@@@@@
		#indexLength = len(y_test)#@@@@@@@@
		#historyList = []#@@@@@@@@
		#while index < indexLength:#@@@@@@@@
			#getSong = y_test[index].split(";")[1]#@@@@@@@@
			#if getSong == "Speech":#@@@@@@@@
				#newYList.append(y_test[index].split(";")[0])#@@@@@@@@
				#newXList.append(X_test[index])#@@@@@@@@
			#index += 1#@@@@@@@@

		# Cleaning the Train Emotions
		#index = 0#@@@@@@@@
		#indexLength = len(y_train)#@@@@@@@@
		#newYTrain = []#@@@@@@@@
		#while index < indexLength:#@@@@@@@@
			#newYTrain.append(y_train[index].split(";")[0])#@@@@@@@@
			#index += 1#@@@@@@@@


		# rewriting the old lists
		#y_test = newYList#@@@@@@@@
		#X_test = newXList#@@@@@@@@
		#y_train = newYTrain#@@@@@@@@


		#print("Test: " + str(newYList))
		#print("Train Feature #: " + str(len(newXList)))
		#print("\n\n")
		#print(y_train)
		#print(newYTrain)



		#Filter Train OR Test data based on emotion
		#trainSelectedEmotion = ["disgust","fearful","surprised"]
		#tempEmotionList = []
		#tempDataList = []
		#for filterIndex, emotion in enumerate(y_train):
			#if emotion in trainSelectedEmotion:
				#tempEmotionList.append(y_train[filterIndex])
				#tempDataList.append(X_train[filterIndex])

		#y_test = np.array(tempEmotionList)
		#X_test = np.array(tempDataList)


		print_Excel(X_train.shape[1])
		move_Cursor('column', 1)


		for i in range(1, amountOfTests + 1): # Starting from 1 and excluding final number
			print("===================================================")
			print("Test :",i," out of ",amountOfTests)
			# best model, determined by a grid search
			model_params = {
				'alpha': 0.01,
				'batch_size': 256,
				'epsilon': 1e-08, 
				'hidden_layer_sizes': (300,), 
				'learning_rate': 'adaptive', 
				'max_iter': 1000, 
			}

			# initialize Multi Layer Perceptron classifier
			# with best parameters ( so far )
			if models == "MLP":
				model = MLPClassifier(**model_params) #MLP / Neural Net
			if models == "KNN":
				model = KNeighborsClassifier(n_neighbors=1, weights='uniform') #KNN
			if models == "SVML":
				model = svm.SVC(kernel='linear',max_iter=-1) # Support Vector Machine Linear Kernel
			if models == "SVMR":
				model = svm.SVC(kernel='rbf',max_iter=-1) # Support Vector Machine Radial Basis Function Kernel
			if models == "SVMP":
				model = svm.SVC(kernel='poly',max_iter=-1) # Support Vector Machine Poly Kernel
			if models == "Random Forest":
				model = RandomForestClassifier() #Random Forest
			if models == "Naive Bayes":
				model = GaussianNB() #Naive Bayes

			# train the model
			#t0 = time.time()
			print("[*] Training the model...")
			#X_train, X_test, y_train, y_test = train_test_split(np.array(knnX), knny, test_size=accuracySplit[1]/100) # THIS WAS FOR KNN
			model.fit(X_train, y_train)

			scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
			print(scores)
			print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

			# predict 25% of data to measure how good we are
			y_pred = model.predict(X_test)

			# calculate the accuracy
			accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
			#t1 = time.time()
			print("Accuracy: {:.2f}%".format(accuracy*100))
			#total = t1-t0
			#print("Duration: " + str(round(total)))
			print(classification_report(y_test, y_pred))
			print(confusion_matrix(y_test, y_pred))


			tempActualPositive, tempActualNegative, tempTruePositive, tempFalsePositive, tempTrueNegative, tempFalseNegative = aggregated_confusion_matrix(y_test, y_pred, selectedEmotions)
			ActualPositive += tempActualPositive
			ActualNegative += tempActualNegative
			TruePositive += tempTruePositive
			FalsePositive += tempFalsePositive
			TrueNegative += tempTrueNegative
			FalseNegative += tempFalseNegative

			

			print("Test :",i," out of ",amountOfTests)


			testResult = accuracy*100
			#if kFoldAccuracy:
			averageAccuracy += scores.mean()*100  # Kfold average
			print_Excel(scores.mean()*100) #KFold Accuracy
			move_Cursor('column', 1)
			#else:EBEN
			averageAccuracy += testResult
			print_Excel(testResult)

			move_Cursor('column', 1)
			
			utilityValue = calculate_utility_value(tempTruePositive, tempFalsePositive, tempTrueNegative, tempFalseNegative)
			print_Excel(utilityValue)
			move_Cursor('column', 1)


			#averageAccuracy += testResult
			########This is for bit 10 tests
			#print_Excel(testResult)
			#move_Cursor('column', 1)
			#########
			out_arr = numpy.array_str(confusion_matrix(y_test, y_pred))
			print_Excel(out_arr)
			#print_Excel(str(confusion_matrix(y_test, y_pred))
			move_Cursor('column', 1)
			#print_Excel(str(round(total)))
			#move_Cursor('column', 1)

			print("===================================================")

		averageAccuracy = averageAccuracy/amountOfTests
		print("AVERAGE: " + str(averageAccuracy) + "\n\n")
		#print_Excel(averageAccuracy)
		#move_Cursor('column', 1)


		# Find highest and lowest
		if averageAccuracy < averageLowest:
			averageLowest = averageAccuracy
		if averageAccuracy > averageHighest:
			averageHighest = averageAccuracy
		averageAccuracy = 0


		ActualPositive = ActualPositive/amountOfTests
		ActualNegative = ActualNegative/amountOfTests
		TruePositive = TruePositive/amountOfTests
		FalsePositive = FalsePositive/amountOfTests
		TrueNegative = TrueNegative/amountOfTests
		FalseNegative = FalseNegative/amountOfTests

		print(str(ActualPositive) + " / " + str(ActualNegative) + " / " + str(TruePositive) + " / " + str(FalsePositive) + " / " + str(TrueNegative) + " / " + str(FalseNegative))
		print("Warn/Warn: " + str(TruePositive))
		print("Warn/Ignore: " + str(FalseNegative))
		print("Ignore/Warn: " + str(FalsePositive))
		print("Ignore/Ignore: " + str(TrueNegative))
		print("\nPositive Support: " + str(ActualPositive) + " / Negative Support: " + str(ActualNegative))
		
		utilityValue = calculate_utility_value(TruePositive, FalsePositive, TrueNegative, FalseNegative)

		print("\nUtility Value: " + str(utilityValue) + "\n")
		print("===================================================")


		print_Excel(str(utilityValue))
		move_Cursor('column', 1)

		print_Excel(str((FalsePositive/ActualNegative)*100))
		#print_Excel("N/A")
		move_Cursor('column', 1)

		print_Excel(str((FalseNegative/ActualPositive)*100))
		#print_Excel("N/A")
		move_Cursor('column', 1)

		print_Excel("WW " + str(TruePositive) + ",WI " + str(FalseNegative) + ",IW " + str(FalsePositive) + ",II " + str(TrueNegative))
		move_Cursor('column', 1)

		print_Excel(str((TruePositive/(TruePositive+FalsePositive)*100)) + " %")
		#print_Excel("N/A")
		move_Cursor('column', 1)

		duration = str(timedelta(seconds=time.time() - t0))
		x = duration.split(':')
		sec = x[2].split('.')
		x[2] = sec[0]
		move_Cursor('column', 1)
		totalTime = 'Current Duration: ' + str(x[0]) + ' H ' + str(x[1]) + ' M ' + str(x[2]) + ' S '
		print_Excel(totalTime)

		new_Row_Left()
		move_Cursor('column', 1)
		wb.save(fileName)
		featureArrayNum += 1

	averageIndentAmount = 3 + len(features) + amountOfTests*2
	move_Cursor('column', averageIndentAmount)

	print_Excel('Highest: ' + str(averageHighest))
	move_Cursor('column', 1)
	print_Excel('Lowest: ' + str(averageLowest))
	move_Cursor('column', 1)

	averageHighest = 0
	averageLowest = 100

	experimentNumber += 1

	new_Row_Left()
	#^^^^^^^^^END OF EXPERIMENT

# <- END OF MODEL

duration = str(timedelta(seconds=time.time() - t0))
x = duration.split(':')
totalTime = 'Total Time: ' + str(x[0]) + ' Hours ' + str(x[1]) + ' Minutes ' + str(x[2]) + ' Seconds '
print(totalTime)

new_Row_Left()
print_Excel(totalTime)
wb.save(fileName)
#originalFileName = path.splitext(fileName)[0]
#duplicateValue = 1
#while path.exists(fileName):
	#fileName = originalFileName + "(" + str(duplicateValue) + ").xls"
	#duplicateValue += 1

#wb.save(fileName)

print("hello")
