import numpy as np
from random import sample
import matplotlib.pyplot as plt

def importData(file):
	"""Imports a dataset."""
	return np.genfromtxt(file, delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

def createLabels(file):
	"""Creates all labels for the data. Instead of using the year, month, day as label, we will generalise to seasons which we do as follows."""
	dates = np.genfromtxt(file, delimiter=';', usecols=[0])
	labels = []
	for label in dates:
		if label < 20000301:
			labels.append('winter')
		elif 20000301 <= label < 20000601:
			labels.append('lente')
		elif 20000601 <= label < 20000901:
			labels.append('zomer')
		elif 20000901 <= label < 20001201:
			labels.append('herfst')
		else: # from 01-12 to end of year
			labels.append('winter')
	return labels

def normalizeData(data):
	"""Normalizes a dataset using the min-max formula."""
	min = np.min(data)
	max = np.max(data)

	normalizedData = [(x - min) / (max - min) for x in data]
	return normalizedData


def centroidAvg(centroidList, clusterList):
	"""This function computes the average (mean) of datapoints associated with each centroid.
	It returns a list containing the mean values of the datapoints associated with each centroid"""
	meanCentroidList = []
	for centroid in centroidList:
		clusterTotal = []
		for item in clusterList:
			if centroid is item[1]:
				clusterTotal.append(item[0])
		meanCentroidList.append(np.mean(clusterTotal))
	return meanCentroidList

def getCentroids(centroidList, dataset, k):
	"""Assigns each data point in the dataset to the closest centroid and returns the updated centroids and the cluster assignments"""
	clusterList = []
	for day in dataset:
		distanceList = []
		for centroid in centroidList:
			distance = np.linalg.norm(day-centroid)
			distanceList.append(distance)
		closestCentroid = distanceList.index(min(distanceList))
		clusterList.append([day, centroidList[closestCentroid]])
	
	return centroidList, clusterList


def kMeans(dataset, k):
	"""Performs a k-means clustering on the given dataset.
	It returns the final centroids and cluster assignements when the centroids are equal to the meancentroids"""
	centroidList = sample(dataset, k)
	while True:
		centroidList, clusterList = getCentroids(centroidList, dataset, k)
		meanCentroidList = centroidAvg(centroidList, clusterList)
		
		if np.array_equal(centroidList, meanCentroidList):
			return centroidList, clusterList
		centroidList = meanCentroidList

def assignLabels(clusterList, labels):
	"""Assigns labels to each datapoint in the clusterlist
	Returns the updated clusterlist with the labels assigned to each datapoint"""
	for index in range(len(clusterList)):
		clusterList[index].append(labels[index])
	
	return clusterList

def countLabels(clusterList, centroidList):
	"""Count the occurrence of each label for each centroid."""
	for centroid in centroidList:
		winterCounter = 0
		springCounter = 0
		summerCounter = 0
		autumnCounter = 0
		for datapoint in clusterList:
			if datapoint[1] == centroid:
				if datapoint[2] == "winter":
					winterCounter += 1
				if datapoint[2] == "lente":
					springCounter += 1
				if datapoint[2] == "zomer":
					summerCounter += 1
				if datapoint[2] == "herfst":
					autumnCounter += 1
		print("Cluster", centroidList.index(centroid) + 1, "bevat", winterCounter, "winters,", springCounter, "lentes,", summerCounter, "zomers en", autumnCounter, "herfsten")

def createSkreePlot(dataset, k):
	"""Creates a scree plot to determine the optimal number of clusters (k) for the k-means algorithm."""
	currentK = []
	averageCentroidDistance = []
	for iteration in range(k):
		iteration += 1
		centroidList, clusterList = kMeans(dataset, iteration)
		totalCentroidDistance = []
		for centroid in centroidList:
			centroidDistance = 0
			for datapoint in clusterList:
				centroidDistance += np.linalg.norm(datapoint[0]-datapoint[1])	
			totalCentroidDistance.append(centroidDistance)
		currentK.append(iteration)
		averageCentroidDistance.append(np.mean(centroidDistance))				

	plt.plot(currentK, averageCentroidDistance)
	plt.xlabel("K")
	plt.ylabel("Gemiddelde centroid afstand")
	plt.show()

def main():
	"""This is the main function :p"""
	dataset = normalizeData(importData('dataset1.csv'))
	datasetLabels = createLabels('dataset1.csv')
	k = 4

	centroidList, clusterList = kMeans(dataset, k)

	clusterList = assignLabels(clusterList, datasetLabels)
	countLabels(clusterList, centroidList)
	
	k = 5
	createSkreePlot(dataset, k)
	k = 10
	createSkreePlot(dataset, k)
	print("------------------------------------------------------------")					 
   
if __name__ == '__main__':
    print("Opdracht 6 - K Means")
    print("------------------------------------------------------------")
    main()
