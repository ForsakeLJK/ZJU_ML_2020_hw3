import numpy as np
import sys
import random

def kmeans(x, k):
	'''
	KMEANS K-Means clustering algorithm

		Input:  x - data point features, n-by-p maxtirx.
				k - the number of clusters

		OUTPUT: idx  - cluster label
				ctrs - cluster centers, K-by-p matrix.
				iter_ctrs - cluster centers of each iteration, (iter, k, p)
						3D matrix.
	'''
	# YOUR CODE HERE
	# begin answer
	N = x.shape[0]

	if k > N:
		sys.exit("k is larger than the number of data points!")
	
	MAX_ITER = 1000

	p = x.shape[1]
	iter_ctrs = np.zeros((1, k, p))
	ctrs = np.zeros((k, p))
	# labels for N samples
	idx = np.zeros(N, dtype=np.int64)

	# choose k centroids randomly
	randCentroidIdx = np.array(random.sample(range(N), k))
	ctrs = x[randCentroidIdx]
	# iter_ctrs[0] = ctrs

	# print(iter_ctrs.shape)

	# print(ctrs)

	for i in range(MAX_ITER):
		# dist[i, j] denotes the distance between center i and sample j
		dist = np.zeros((k, N))

		for j in range(k):
			tmp = x - ctrs[j]
			# square distance
			dist[j] = np.sum(np.square(tmp), axis=1)

		# n labels
		curIdx = np.argmin(dist, axis=0)

		# unchanged clustering, break
		if np.array_equal(curIdx, idx):
			break
		else: # changed clustering, update labels
			idx = curIdx

		# update centroid
		for m in range(k):
			thisClusterIndex = np.where(idx == m)
			# mean of each col
			ctrs[m] = np.mean(x[thisClusterIndex], axis=0)
		
		if i == 0:
			iter_ctrs[0] = ctrs
		else:
			iter_ctrs = np.append(iter_ctrs, np.array([ctrs]), axis=0)
		# print(ctrs)

	# end answer
	# iter_ctrs = np.append(iter_ctrs, np.array([ctrs]), axis=0)
	# print(iter_ctrs.shape)

	# print("iteration: {}".format(iter_ctrs.shape[0]))

	return idx, ctrs, iter_ctrs
