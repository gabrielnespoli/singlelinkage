import numpy as np
import string
import re

INFINITY = 9999
def removePunctuations(s):
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	return regex.sub(' ', s)

def wordcount(book):
	bookF = open(book, mode='r', encoding='utf-8-sig').read()
	open(book, mode='w', encoding='utf-8').write(bookF)
	bookF = open(book)

	d = {}
	for line in bookF:
		line = removePunctuations(line)
		words = line.split()
		for eachWord in words:
			eachWord = eachWord.lower()
			if(eachWord not in d):
				d[eachWord] = 1 
			else:
				d[eachWord] += 1
	return d

def bag(wc, threshold=1):	
	return {k for k, v in wc.items() if v >= threshold}

def jaccard(s1, s2):
	return len(s1.intersection(s2)) / len(s1.union(s2))

# in a matrix of distances, get the smallest distance and the coordinate of it
def small_distance(D):
	N = len(D)
	smallest = D[0,0]
	coord = (0,0)
	for i in range(N):
		for j in range(N):
			if(D[i,j] < smallest):
				smallest = D[i,j]
				coord = (i,j)
	return ((smallest,coord))

# make infinity the distance between each document from itself
def set_equal_infinity(D):
	for i in range(len(D)):
		for j in range(len(D)):
			if(i==j):
				D[i,j] = INFINITY

def initialize_cluster(D):
	cluster = dict()
	for i in range(len(D)):
		cluster[i] = i
	return cluster

def initialize_names(D):
	names = []
	for i in range(len(D)):
		names.append("D"+str(i))
	return np.array(names)

def update_names(names, coord):
	newName = names[coord[0]]+names[coord[1]]
	
	# delete the previous name 
	names = np.delete(names, coord[1])
	
	# concatenate the new name, Ex: D1D2
	l = list(names)
	l[coord[0]] = newName
	return np.array(l)

def update_cluster_sequence(cluster,name):
	# transform a string with a cluster name 'D1D2D2' in numpy array ['D','1','D','2','D','3'], and then get just the numbers [1,2,3]
	name = np.array(list(name))[1::2]
	index = [int(x) for x in name]
	
	#list of index to be updated
	l = []
	for i in index:
		l.append(cluster[i])
	for i in index:
		cluster[i] = min(l)

	return cluster
	
def single_linkage(D_original, k=2):
	D = np.array(D_original)
	N = len(D)
	cluster = initialize_cluster(D)
	names = initialize_names(D)
	set_equal_infinity(D)
	
	for j in range(N-1):
		if(len(np.unique(np.array(list(cluster.values())))) > k):
			dist,coord = small_distance(D)
			
			# put in the same cluster two documents that are close to each other
			names = update_names(names, coord)

			# set the cluster number of new document as the sequence of the old documents in the cluster
			cluster = update_cluster_sequence(cluster,names[coord[0]])

			# get the column and line of the documents that are going to be clustered
			clusteredColumn = D[:,(coord[0],coord[1])]
			
			# re-calculate the distances between the other documents and this cluster
			arrayMinimums = np.zeros(len(clusteredColumn))
			for i in range(len(clusteredColumn)):
				if(clusteredColumn[i][0] == INFINITY or clusteredColumn[i][1] == INFINITY):
					arrayMinimums[i] = INFINITY
				elif(clusteredColumn[i][0] > clusteredColumn[i][1]):
					arrayMinimums[i] = clusteredColumn[i][1]
				else:
					arrayMinimums[i] = clusteredColumn[i][0]
	
			arrayMinimums = arrayMinimums.reshape(len(clusteredColumn),1)
			
			# delete the two column distances of the two documents to be clustered
			D = np.delete(D, [coord[0],coord[1]], 1)

			# concatenate the column of minimums
			D = np.insert(D, coord[0], arrayMinimums.T,axis=1)

			# i've changed the columns, now i'm changing the line
			clusteredLine = D[(coord[0],coord[1]),:]
			arrayMinimums = np.zeros(len(clusteredLine[0]))
			for i in range(len(clusteredLine[0])):
				if(clusteredLine[0][i] == INFINITY or clusteredLine[1][i] == INFINITY):
					arrayMinimums[i] = INFINITY
				elif(clusteredLine[0][i] > clusteredLine[1][i]):
					arrayMinimums[i] = clusteredLine[1][i]
				else:
					arrayMinimums[i] = clusteredLine[0][i]
	
			# delete the two line distances of the two documents to be clustered
			D = np.delete(D, [coord[0],coord[1]], axis=0)
	
			#concatenate the line of minimums
			D = np.insert(D, coord[0], arrayMinimums,axis=0)

	return list(cluster.values())
