import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import math

#load csv files into DataFrames
testData = pd.read_csv('test.csv', header = 0)
trainData = pd.read_csv('train.csv', header = 0)

#create ndarray of train data
trainX = trainData.values

#create ndarray of train labels
trainY = trainX[:, 1]

#reformat data
#remove passenger num and survived cols from trainX
trainX = np.delete(trainX, np.s_[0:2:], 1)
#remove name col
trainX = np.delete(trainX, np.s_[1:2:], 1)
#turn trainX into an ndarray
trainX = np.asarray(trainX)
#reformat data into floats and find average age by gender
numAgesFemales = 0
numAgesMales = 0
totalAgeFemales = 0
totalAgeMales = 0
for i in range(0, 891): #for every person
    if trainX[i, 1] == "female":
        trainX[i, 1] = 0 #reformat gender data
        if not math.isnan(trainX[i, 2]): #if age exists
            totalAgeFemales += trainX[i, 2] #add to total age and numAges
            numAgesFemales += 1
    else:
        trainX[i, 1] = 1#reformat gender data
        if not math.isnan(trainX[i, 2]): #if age exists
            totalAgeMales += trainX[i, 2] #add to total age and numAges
            numAgesMales += 1
#find average age for men and women
avgAgeFemales = totalAgeFemales/numAgesFemales
avgAgeMales = totalAgeMales/numAgesMales
#print(str(numAgesFemales) + " females are average " + str(avgAgeFemales) + "yrs old")
#print(str(numAgesMales) + " males are average " + str(avgAgeMales) + "yrs old")
#
#replace missing ages with average age of that gender on board
for i in range(0, 891): #for every person
    if math.isnan(trainX[i, 2]): #if no age is listed
        if trainX[i, 1] == 0:#if female
            trainX[i, 2] = avgAgeFemales#replace with avg
        else:#if male
           trainX[i, 2] = avgAgeMales#replace with avg
#reformat Embarked col
for i in range(0, 891): #for every person
    if trainX[i, 8] == "S":
        trainX[i, 8] = 0.0
    elif trainX[i, 8] == "Q":
        trainX[i, 8] = 1.0
    else:
        trainX[i, 8] = 2.0
#remove ticket and cabin columns
trainX = np.delete(trainX, np.s_[5:6:], 1)
trainX = np.delete(trainX, np.s_[6:7:], 1)

##print sample of data
#for i in range (0, 20):
#    for j in range(0, 7):
#        print(str(trainX[i, j]) + ", ", end = "")
#    print(" ")

#reformat trainY
trainY = np.asarray(trainY)
trainY = trainY.astype('int')



#reformat SUBMISSION data
testX = testData.values
#remove passenger num and survived cols from testX
testX = np.delete(testX, np.s_[0:1:], 1)
#remove name col
testX = np.delete(testX, np.s_[1:2:], 1)
#turn testX into an ndarray
testX = np.asarray(testX)
#reformat data into floats and find average age by gender
for i in range(0, 418): #for every person
    if testX[i, 1] == "female":
        testX[i, 1] = 0 #reformat gender data
    else:
        testX[i, 1] = 1#reformat gender data
#replace missing ages with average age of that gender on board
for i in range(0, 418): #for every person
    if math.isnan(testX[i, 2]): #if no age is listed
        if testX[i, 1] == 0:#if female
            testX[i, 2] = avgAgeFemales#replace with avg
        else:#if male
           testX[i, 2] = avgAgeMales#replace with avg
#reformat Embarked col
for i in range(0, 418): #for every person
    if testX[i, 8] == "S":
        testX[i, 8] = 0.0
    elif testX[i, 8] == "Q":
        testX[i, 8] = 1.0
    else:
        testX[i, 8] = 2.0
#remove ticket and cabin columns
testX = np.delete(testX, np.s_[5:6:], 1)
testX = np.delete(testX, np.s_[6:7:], 1)
#check for remaining missing data, replace with 0 since it is very rare
for i in range(0, 418): #for every person
    for j in range(0, 6): #for every column
        if math.isnan(testX[i, j]): #if is nan
            testX[i, j] = 0 #replace with 0

#print sample  of data
for i in range (0, 20):
    for j in range(0, 7):
        print(str(testX[i, j]) + ", ", end = "")
    print(" ")

##split data into train and test sets
#trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size = .4)

#create classifiers
my_classifier = tree.DecisionTreeClassifier()
#and
my_kClassifier = KNeighborsClassifier()
#and 
my_randForest = RandomForestClassifier()

#train classifiers
my_classifier.fit(trainX, trainY)
my_kClassifier.fit(trainX, trainY)
my_randForest.fit(trainX, trainY)

##make predictions
#predictions = my_classifier.predict(testX)
#kpredictions = my_kClassifier.predict(testX)
#randPredictions = my_randForest.predict(testX)

##compare predictions to correct answers 
#print("decision tree score: " + str(accuracy_score(testY, predictions)))
#print("kneighbors score: " + str(accuracy_score(testY, kpredictions)))
#print("rand forest score: " + str(accuracy_score(testY, randPredictions)))

##check for infinite vals/nan
#print(str(np.any(np.isnan(testX))))
#print(str(np.all(np.isfinite(testX))))

#make submission predictions
subPredictions = my_classifier.predict(testX)
subKpredictions = my_kClassifier.predict(testX)
subRandPredictions = my_randForest.predict(testX)

print(str(subRandPredictions))

##make a DataFrame of submission data
#make passengerIDNum array
passNum = np.ndarray(shape=(418, 1), dtype=int)
for i in range(0, 418):
    passNum[i] = i+892
#reformat subRandPredictions into a 2D ndarray
subRandPredictions2D = np.ndarray(shape=(418, 1), dtype=int)
for i in range(0, 418):
    subRandPredictions2D[i] = subRandPredictions[i]
#concatenate IDNum and predictions
print(str(subRandPredictions.ndim) + ", " + str(passNum.ndim))
submissionRandF = np.concatenate((passNum, subRandPredictions2D), axis = 1)
#create DataFrame
DfRandF = pd.DataFrame(data = submissionRandF, columns=["PassengerId", "Survived"], index = None, dtype=int)
#transfer DataFrame to CSV
#DfRandF.to_csv(path_or_buf="RandomForestSubmission2.csv")