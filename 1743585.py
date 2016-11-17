import sys
import lib1743585 as mylib
import numpy as np
import collections

def jaccard_table_distances(b):
	size = len(b)
	t = np.zeros(size*size).reshape(size,size)
	for i in range(size):
		book1 = b[i]
		for j in range(size):
			book2 = b[j]
			t[i,j] = 1 - mylib.jaccard(book1,book2)
	return t

def output(bags,jaccard_table,clusters):
	print(sorted([len(x) for x in bags], reverse=True))	
	np.set_printoptions(precision=2)
	print(jaccard_table)
	print(jaccard_table.sum()/len(jaccard_table)**2)

	d = collections.OrderedDict()
	for e in clusters:
		same_cluster = []
		for i in range(len(clusters)):
			if(e == clusters[i]):
				same_cluster.append(i)
		d[str(same_cluster)] = same_cluster
	print(list(d.values()))
	
def main():
	bags = []
	for bookF in sys.argv[1:]:
		bags.append(mylib.bag(mylib.wordcount(bookF),10))
	
	jaccard_table = jaccard_table_distances(bags)
	clusters = mylib.single_linkage(jaccard_table,3)
	output(bags,jaccard_table,clusters)

if __name__ == "__main__":
    main()
	
