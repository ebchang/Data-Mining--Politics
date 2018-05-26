# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:03:56 2015

@author: evanchang
"""
import csv
import random
import math
import matplotlib.pyplot as plt
import numpy as np
 
##def loadCsv(filename):
##	lines = csv.reader(open(filename, "rb"))
##	dataset = list(lines)
##	for i in range(len(dataset)):
##		dataset[i] = [float(x) for x in dataset[i]]
##	return dataset


filename = 'File Location'

def loadCsv(filename):
    raw = open(filename)
    hold = raw.readline()
    dataset = raw.readlines()
    id_hold = {}
    name_list =[]
    feature_list = []
#    class_list = []
    name_class_dict ={}
    raw.close()
    for i in range(len(dataset)):
        dataset[i] = dataset[i].split(',')
    for i in range(len(dataset)):
        a, b, *rest = dataset[i]
        features = rest
        feature_list.append(features)
        hold= str(a) + str(b)
        name_list.append(hold)
    for i in range(len(feature_list)):
        feature_list[i] = [float(x) for x in feature_list[i]]
#        class_list.append(rest[-1])
    return  feature_list

def discretize (dat):
    mins = [99999] * len(dat[0]) # adjust if necessary
    maxs = [-99999] * len(dat[0]) # adjust if necessary
    for datum in dat:
        for i in range(len(datum)):
            if datum[i] < mins[i]:
                mins[i] = datum[i]
            if datum[i] > maxs[i]:
                maxs[i] = datum[i]
    for i in range(len(mins)):
        if mins[i] != 0 or maxs[i] != 1: 
            mid = (mins[i] + maxs[i]) / float(2)
            for j in range(len(dat)):
                if dat[j][i] < mid:
                    dat[j][i] = 0
                else:
                    dat[j][i] = 1
    return dat 
    
dataset = discretize(loadCsv(filename))

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
 
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated
 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
 
def summarize(dataset):
        summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries
 
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries
 
def calculateProbability(x, mean, stdev):
        x += 1 #Add one to prevent division by zero error
        mean +=1
        stdev +=1
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
	correct = 0
	#true_value = []
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
			#true_value.append(testSet[i][-1])
	return (correct/float(len(testSet))) * 100.0

def get_true_pred(testSet, predictions):
    true_value = []
    hold_list = []
    for i in range(len(testSet)):
        true_value.append(testSet[i][-1])
    for j in range(len(testSet)):
        hold_list.append((testSet[j][-1], predictions[j]))
    return hold_list

def ROC(filename):
    dataset = loadCsv(filename)
    splitRatio = 0.67
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    summaries = summarizeByClass(trainingSet)
    predictions = getPredictions(summaries, testSet)
    ROC_list = get_true_pred(testSet, predictions)
    return ROC_list

ROC_list = ROC(filename)

def counter(ROC_list):
    counter_positive = 0
    counter_negative = 0
    FN =  0
    FP = 0
    wrong_list = []
    for i in range(len(ROC_list)):
        if ROC_list[i][0] == 1.0 and ROC_list[i][1] == 1.0:
            counter_positive += 1
        elif ROC_list[i][0] == 0.0 and ROC_list[i][1] == 0.0:
            counter_negative += 1
        elif ROC_list[i][0] == 1.0 and ROC_list[i][1] ==0.0:
            FN +=1
            wrong_list.append((ROC_list[i][0],ROC_list[i][1]))
        else:
            FP += 1
            wrong_list.append((ROC_list[i][0],ROC_list[i][1]))
#    return counter_positive, counter_negative, wrong, wrong_list

    x = FP /(FP + counter_negative)
    y = counter_positive / (counter_positive + FN)

    plt.plot(x,y)
    plt.show()

#    auc = np.trapz(y,x)
    return y, x
    
#plt.plot(counter(ROC_list))
#plt.show()
auc = np.trapz(counter(ROC_list))
    
#filename = "/Users/evanchang/Desktop/dat.csv"
def main(filename):
	dataset = loadCsv(filename)
	splitRatio = 0.67
#	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	#print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	#print(summaries)
	#print(len(summaries))
	# test model
	predictions = getPredictions(summaries, testSet)
	#print(predictions)
	#print(len(predictions))
	accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: {0}%').format(accuracy)
	print(accuracy)
 
#main(filename)
